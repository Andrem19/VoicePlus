#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, asyncio, json, signal, datetime as dt, subprocess, contextlib, time, re as _re
import websockets

# ====== (опционально красиво) rich ======
HAVE_RICH = True
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except Exception:
    HAVE_RICH = False

# ------------------- конфиг / окружение -------------------
def load_env(path=".env"):
    cfg = {}
    if os.path.exists(path):
        for line in open(path, "r", encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg

ENV = load_env(".env")

API_KEY = ENV.get("SPEECHMATICS_API") or ENV.get("SPEECHMATICS_API_KEY") or ""
if not API_KEY:
    print("ERROR: Добавьте SPEECHMATICS_API=... в .env", file=sys.stderr)
    sys.exit(1)

DEBUG  = ENV.get("DEBUG", "0").lower() in ("1","true","yes")
SM_ENDPOINT = ENV.get("SPEECHMATICS_WSS", "wss://eu2.rt.speechmatics.com/v2/")

SAMPLE_RATE = int(ENV.get("SAMPLE_RATE", "16000"))
ENCODING = "pcm_s16le"
MAX_DELAY = float(ENV.get("MAX_DELAY", "1.3"))
MAX_DELAY_MODE = ENV.get("MAX_DELAY_MODE", "flexible")
EOU_SILENCE = float(ENV.get("EOU_SILENCE", "0.8"))

# live-троттлинг/эвристики
PARTIAL_DEBOUNCE_MS = int(ENV.get("PARTIAL_DEBOUNCE_MS", "300"))
MIN_DELTA_CHARS     = int(ENV.get("MIN_DELTA_CHARS", "8"))
MIN_WORDS_PARTIAL   = int(ENV.get("MIN_WORDS_PARTIAL", "2"))
SILENCE_FLUSH_MS    = int(ENV.get("SILENCE_FLUSH_MS", "700"))
EOU_FINALIZE_MS     = int(ENV.get("EOU_FINALIZE_MS", "1100"))

# перевод (только финалы)
ENABLE_TRANSLATION     = ENV.get("ENABLE_TRANSLATION", "0").lower() in ("1","true","yes")
TRANSLATE_TO           = (ENV.get("TRANSLATE_TO", "ru") or "ru").strip().lower()
TRANSLATION_WINDOW_MS  = int(ENV.get("TRANSLATION_WINDOW_MS", "2000"))

PRINT_AUDIOADDED    = ENV.get("PRINT_AUDIOADDED", "0").lower() in ("1","true","yes")

# управление захватом линий
CAPTURE_S1 = ENV.get("CAPTURE_S1", "1").lower() in ("1","true","yes")
CAPTURE_S2 = ENV.get("CAPTURE_S2", "1").lower() in ("1","true","yes")

# логи
LOG_DIR = os.path.expanduser("call_logs"); os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"call_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
def ts(): return dt.datetime.now().strftime("%H:%M:%S")

# ------------------- вывод (цвета / утилиты) -------------------
def _supports_color():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
USE_COLOR = _supports_color()
RESET = "\033[0m" if USE_COLOR else ""
DIM   = "\033[2m" if USE_COLOR else ""
RED   = "\033[31m" if USE_COLOR else ""
GRN   = "\033[32m" if USE_COLOR else ""
CYAN  = "\033[36m" if USE_COLOR else ""
ITAL  = "\033[3m"  if USE_COLOR else ""
ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")
def strip_ansi(s: str) -> str: return ANSI_RE.sub("", s)
_print_lock = asyncio.Lock()

async def print_and_log(line: str):
    async with _print_lock:
        if LIVE and LIVE.use_rich and LIVE.console:
            LIVE.console.print(line)
        else:
            print(line, flush=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(strip_ansi(line) + "\n")

async def info_line(s: str):
    if DEBUG:
        await print_and_log(f"{DIM}{s}{RESET}" if not (LIVE and LIVE.use_rich) else s)

async def head_line(s: str):
    await print_and_log(f"{CYAN}{s}{RESET}" if not (LIVE and LIVE.use_rich) else s)

# ------------------- Live UI -------------------
class LiveManager:
    """Две «живые» строки (S1/S2) в футере. История печатается отдельно."""
    def __init__(self, use_rich: bool):
        self.use_rich = HAVE_RICH and use_rich and sys.stdout.isatty()
        self.console = Console(highlight=False, soft_wrap=False) if HAVE_RICH else None
        self.live = None
        self.active = False
        self._footer_state = {"S1":"", "S2":""}
        self._footer_buffer = {"S1":"", "S2":""}
        self._footer_last_seen_ms = {"S1":0.0, "S2":0.0}
        self._footer_last_render_ms = {"S1":0.0, "S2":0.0}
        self._flusher_task = None

    def _render_footer(self):
        s1 = self._footer_state["S1"] or "…"
        s2 = self._footer_state["S2"] or "…"
        if HAVE_RICH:
            table = Table.grid(padding=(0,1))
            table.add_row(Text("LIVE S1:", style="bold red"),   Text(s1))
            table.add_row(Text("LIVE S2:", style="bold green"), Text(s2))
            return Panel(table, title="Live", border_style="cyan")
        return f"[LIVE]\nS1: {s1}\nS2: {s2}\n"

    async def start(self):
        self.active = True
        if self.use_rich:
            self.live = Live(self._render_footer(), console=self.console, refresh_per_second=8, transient=False)
            self.live.start()
        self._flusher_task = asyncio.create_task(self._silence_flusher())

    async def stop(self):
        self.active = False
        if self._flusher_task:
            self._flusher_task.cancel()
            with contextlib.suppress(Exception): await self._flusher_task
        if self.use_rich and self.live:
            self.live.stop()
            self.live = None

    async def _update_footer_now(self):
        if not self.active:
            return
        if self.use_rich and self.live:
            self.live.update(self._render_footer())
        else:
            print(f"{DIM}[LIVE] S1: {self._footer_state['S1'] or '…'} | S2: {self._footer_state['S2'] or '…'}{RESET}", flush=True)

    async def _silence_flusher(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                now_ms = time.monotonic() * 1000.0
                for label in ("S1","S2"):
                    buf = self._footer_buffer[label]
                    seen = self._footer_last_seen_ms[label]
                    rend = self._footer_last_render_ms[label]
                    if not buf:
                        continue
                    if buf != self._footer_state[label] and (now_ms - seen) >= SILENCE_FLUSH_MS and (now_ms - rend) >= 50:
                        self._footer_state[label] = buf
                        self._footer_last_render_ms[label] = now_ms
                        await self._update_footer_now()
        except asyncio.CancelledError:
            pass

    async def show_live_text(self, label: str, text: str):
        now_ms = time.monotonic() * 1000.0
        last_drawn = self._footer_state[label]
        grew = (len(text) - len(last_drawn)) >= MIN_DELTA_CHARS
        punct = bool(_re.search(r"[.!?…]$", text))
        debounce_ok = (now_ms - self._footer_last_render_ms[label]) >= PARTIAL_DEBOUNCE_MS

        self._footer_buffer[label] = text
        self._footer_last_seen_ms[label] = now_ms

        if (grew and debounce_ok) or punct:
            self._footer_state[label] = text
            self._footer_last_render_ms[label] = now_ms
            await self._update_footer_now()

    async def clear_live_line(self, label: str):
        self._footer_state[label] = ""
        self._footer_buffer[label] = ""
        self._footer_last_render_ms[label] = time.monotonic() * 1000.0
        await self._update_footer_now()

    async def print_final_with_extras(self, label: str, text: str, tr_text: str | None = None):
        await print_and_log(f"{ts()} | {label}: {text}")
        if ENABLE_TRANSLATION:
            if tr_text and tr_text.strip():
                await print_and_log(f"{ts()} | {label} → RU: {tr_text.strip()}")
            else:
                await print_and_log(f"{ts()} | {label} → RU: —")
        await print_and_log(f"{ts()} | SUGGEST: —")

# ------------------- менеджер перевода -------------------
class TranslationManager:
    """
    Копит финальные AddTranslation и печатает их пост-фактум, если не успели к моменту фиксации реплики.
    Ничего не блокирует.
    """
    def __init__(self, live: LiveManager):
        self.live = live
        self._parts = {"S1": [], "S2": []}
        self._pending = {"S1": None, "S2": None}  # {"deadline_ms":..., "printed":bool}
        self._watch_task = None

    def _joined(self, label: str) -> str:
        txt = " ".join(p.strip() for p in self._parts[label] if p and p.strip())
        return _re.sub(r"\s+", " ", txt).strip()

    async def start(self):
        if ENABLE_TRANSLATION:
            self._watch_task = asyncio.create_task(self._watchdog())

    async def stop(self):
        if self._watch_task:
            self._watch_task.cancel()
            with contextlib.suppress(Exception): await self._watch_task

    async def on_translation_chunk(self, label: str, text: str):
        if not ENABLE_TRANSLATION or not text:
            return
        self._parts[label].append(text)
        pend = self._pending[label]
        if pend and not pend["printed"] and (time.monotonic() * 1000.0) <= pend["deadline_ms"]:
            tr = self._joined(label)
            if tr:
                await print_and_log(f"{ts()} | {label} → RU: {tr}")
                pend["printed"] = True
                self._parts[label].clear()

    async def on_utterance_finalized(self, label: str) -> str | None:
        if not ENABLE_TRANSLATION:
            return None
        now_ms = time.monotonic() * 1000.0
        tr_now = self._joined(label)
        if tr_now:
            self._parts[label].clear()
            self._pending[label] = None
            return tr_now
        self._pending[label] = {"deadline_ms": now_ms + TRANSLATION_WINDOW_MS, "printed": False}
        return None

    async def on_new_utterance_started(self, label: str):
        self._pending[label] = None
        self._parts[label].clear()

    async def _watchdog(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                now_ms = time.monotonic() * 1000.0
                for label in ("S1","S2"):
                    pend = self._pending[label]
                    if not pend:
                        continue
                    if pend["printed"] or now_ms > pend["deadline_ms"]:
                        self._pending[label] = None
                        self._parts[label].clear()
        except asyncio.CancelledError:
            pass

LIVE: LiveManager | None = None
TM: TranslationManager | None = None

# ------------------- Коалесцер реплик -------------------
class Coalescer:
    def __init__(self, live: LiveManager, tm: TranslationManager | None):
        self.live = live
        self.tm = tm
        self._parts = {"S1":[], "S2":[]}
        self._partial = {"S1":"", "S2":""}
        self._last_activity = {"S1":0.0, "S2":0.0}
        self._watch_task = None

    def _joined(self, label: str) -> str:
        parts = self._parts[label][:]
        if self._partial[label]:
            parts.append(self._partial[label])
        txt = " ".join(p.strip() for p in parts if p.strip())
        return _re.sub(r"\s+", " ", txt).strip()

    async def start(self):
        self._watch_task = asyncio.create_task(self._watchdog())

    async def stop(self):
        if self._watch_task:
            self._watch_task.cancel()
            with contextlib.suppress(Exception): await self._watch_task
        for label in ("S1","S2"):
            await self._finalize_if_needed(label, force=True)

    async def on_partial(self, label: str, text: str):
        if not self._parts[label] and not self._partial[label]:
            if TM:
                await TM.on_new_utterance_started(label)
        self._partial[label] = text or ""
        self._last_activity[label] = time.monotonic() * 1000.0
        await self.live.show_live_text(label, self._joined(label))

    async def on_final_chunk(self, label: str, text: str):
        if text:
            self._parts[label].append(text)
        self._partial[label] = ""
        self._last_activity[label] = time.monotonic() * 1000.0
        await self.live.show_live_text(label, self._joined(label))

    async def _watchdog(self):
        try:
            while True:
                await asyncio.sleep(0.05)
                now_ms = time.monotonic() * 1000.0
                for label in ("S1","S2"):
                    last = self._last_activity[label]
                    if last <= 0:
                        continue
                    if (now_ms - last) >= EOU_FINALIZE_MS:
                        await self._finalize_if_needed(label, force=False)
        except asyncio.CancelledError:
            pass

    async def _finalize_if_needed(self, label: str, force: bool):
        txt = self._joined(label)
        if not txt:
            return
        tr_now = None
        if TM:
            tr_now = await TM.on_utterance_finalized(label)
        await self.live.clear_live_line(label)
        await self.live.print_final_with_extras(label, txt, tr_text=tr_now)
        self._parts[label].clear()
        self._partial[label] = ""
        self._last_activity[label] = 0.0

# ------------------- PipeWire helpers -------------------
def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8","ignore")
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8","ignore")
    except Exception:
        return ""

def list_node_names():
    out = run_cmd(["wpctl","status","--name"])
    return _re.findall(r'([A-Za-z0-9_.:-]+)\s*$', out, flags=_re.M)

def find_bt_sink_any():
    for n in list_node_names():
        if n.startswith("bluez_output."):
            return n
    return ""

def find_alsa_sink_any():
    for n in list_node_names():
        if n.startswith("alsa_output."):
            return n
    return ""

def get_default_sink_name():
    out = run_cmd(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"])
    m = _re.search(r'node\.name\s*=\s*"([^"]+)"', out)
    return m.group(1) if m else ""

def find_hyperx_mic():
    names = list_node_names()
    for n in names:
        if n.startswith("alsa_input.") and ("HyperX" in n or "DuoCast" in n or "usb-" in n):
            return n
    for n in names:
        if n.startswith("alsa_input."): return n
    return ""

async def wait_for_sink(mode_or_name: str, timeout_sec=180):
    t0 = dt.datetime.now()
    def pick():
        if mode_or_name == "AUTO_DEFAULT_SINK":
            return get_default_sink_name() or ""
        if mode_or_name == "AUTO_BT_SINK":
            return find_bt_sink_any()
        if mode_or_name == "AUTO_ALSA_SINK":
            return find_alsa_sink_any()
        names = list_node_names()
        return mode_or_name if mode_or_name and mode_or_name in names else ""
    n0 = pick()
    if n0:
        return n0
    while True:
        cand = pick()
        if cand:
            return cand
        if (dt.datetime.now() - t0).total_seconds() > timeout_sec:
            return ""
        await asyncio.sleep(0.3)

# --- Инициализация устройств S1/S2 ---
S1_DEVICE = ENV.get("S1_DEVICE") or find_hyperx_mic()
S2_TARGET = ENV.get("S2_TARGET") or ""

# ------------------- Speechmatics helpers -------------------
async def connect_ws(url, headers):
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)

def _text_from_transcript_results(d: dict) -> str:
    words = []
    for r in d.get("results", []):
        alts = r.get("alternatives", [])
        if alts:
            w = alts[0].get("content", "")
            if w: words.append(w)
    return " ".join(words).strip()

def _extract_transcript(d: dict) -> str:
    meta_t = (d.get("metadata") or {}).get("transcript") or ""
    res_t  = _text_from_transcript_results(d)
    return res_t if len(res_t) > len(meta_t) else meta_t

def _extract_translation(d: dict) -> str:
    # либо строкой, либо через results[].content
    if isinstance(d.get("translation"), str) and d["translation"].strip():
        return d["translation"].strip()
    meta = d.get("metadata") or {}
    if isinstance(meta.get("translation"), str) and meta["translation"].strip():
        return meta["translation"].strip()
    words = []
    for r in d.get("results", []):
        c = r.get("content", "")
        if c:
            words.append(c)
    return " ".join(words).strip()

async def start_sm(label: str, want_translation: bool):
    headers = [("Authorization", f"Bearer {API_KEY}")]
    ws = await connect_ws(SM_ENDPOINT, headers)

    transcription_config = {
        "language": "en",
        "output_locale": "en-GB",
        "enable_partials": True,
        "max_delay": MAX_DELAY,
        "max_delay_mode": MAX_DELAY_MODE,
        "operating_point": "enhanced",
        "conversation_config": {"end_of_utterance_silence_trigger": EOU_SILENCE},
        "audio_filtering_config": {"volume_threshold": 0},
    }

    payload = {
        "message": "StartRecognition",
        "audio_format": {"type": "raw", "encoding": ENCODING, "sample_rate": SAMPLE_RATE},
        "transcription_config": transcription_config
    }

    # ВАЖНО: translation_config — на верхнем уровне, а не внутри transcription_config
    if want_translation:
        payload["translation_config"] = {
            "target_languages": [TRANSLATE_TO],
            "enable_partials": False
        }

    await ws.send(json.dumps(payload))

    recognition_started = asyncio.Event()
    first_error = {"type": None, "reason": None}

    async def reader():
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    await info_line(f"[{label}] [raw] {msg[:200]}")
                    continue

                m = data.get("message")

                if m == "RecognitionStarted":
                    await info_line(f"[{label}] RecognitionStarted")
                    recognition_started.set()

                elif m in ("AddPartialTranscript", "AddTranscript"):
                    txt = _extract_transcript(data)
                    if not txt:
                        continue
                    if m == "AddPartialTranscript":
                        await COALESCE.on_partial(label, txt)
                    else:
                        await COALESCE.on_final_chunk(label, txt)

                elif m == "AddTranslation":
                    tr = _extract_translation(data)
                    if tr and TM:
                        await TM.on_translation_chunk(label, tr)

                elif m == "AddPartialTranslation":
                    # отключены (мы не просим partial), игнор
                    pass

                elif m == "Error":
                    first_error["type"] = data.get("type")
                    first_error["reason"] = data.get("reason")
                    await info_line(f"[{label}] Error: {data}")

                elif m == "AudioAdded":
                    if DEBUG and PRINT_AUDIOADDED:
                        await info_line(f"[{label}] {m}: {data}")

                else:
                    await info_line(f"[{label}] {m or 'UNK'}: {data}")

        except websockets.ConnectionClosed as e:
            await info_line(f"[{label}] WS closed: {e.code} {e.reason}")
        except Exception as e:
            await info_line(f"[{label}] reader exception: {e}")

    async def send_audio_chunk(chunk: bytes):
        await ws.send(chunk)

    async def finish():
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))
        with contextlib.suppress(Exception):
            await ws.close()
        # дать таймауту закрытия освободить квоту и транспорт
        await asyncio.sleep(0.3)

    return ws, reader, send_audio_chunk, finish, recognition_started, first_error

# ------------------- захват -------------------
async def run_pw_record(cmd, on_chunk, label):
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await info_line(f"[{label}] started: {' '.join(cmd)}")

    last_t = time.monotonic()
    _bytes_box = [0]

    async def pump_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line: break
            s = line.decode("utf-8","ignore").rstrip()
            if s: await info_line(f"[{label}][pw-record] {s}")

    async def stats2():
        nonlocal last_t
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            dt_s = max(1e-6, now - last_t)
            kbps = (_bytes_box[0] / 1024.0) / dt_s
            await info_line(f"[{label}] tx ~{kbps:.1f} KB/s (last {dt_s:.2f}s, chunk { _bytes_box[0]//1024 } KB)")
            _bytes_box[0] = 0
            last_t = now

    t_err = asyncio.create_task(pump_stderr())
    t_stat = asyncio.create_task(stats2())

    try:
        while True:
            chunk = await proc.stdout.read(3200)  # ~100мс @ 16kHz mono s16
            if not chunk: break
            _bytes_box[0] += len(chunk)
            await on_chunk(chunk)
    finally:
        with contextlib.suppress(ProcessLookupError): proc.terminate()
        with contextlib.suppress(Exception): await t_err
        with contextlib.suppress(Exception): t_stat.cancel()
        await info_line(f"[{label}] stopped")

# ------------------- утилиты старта/фоллбэка -------------------
async def _wait_start_or_error(label, started_evt, err_box, timeout=8.0):
    t0 = time.monotonic()
    while True:
        if started_evt.is_set():
            return "started", None
        if err_box.get("type"):
            return "error", (err_box.get("type"), err_box.get("reason"))
        if (time.monotonic() - t0) > timeout:
            return "timeout", None
        await asyncio.sleep(0.05)

def _is_translation_schema_error(err_type: str, reason: str) -> bool:
    if not reason:
        return False
    r = reason.lower()
    return ("translation_config" in r and "not allowed" in r) or ("translation" in r and "invalid" in r)

async def _close_reader_and_ws(sess, reader_task):
    # sess: (name, ws, r, send, fin, started, err)
    with contextlib.suppress(Exception):
        await sess[4]()  # fin()
    with contextlib.suppress(Exception):
        reader_task.cancel()
        await asyncio.sleep(0)  # дать таску отмениться

# ------------------- main -------------------
COALESCE: Coalescer | None = None
LIVE: LiveManager | None = None
TM: TranslationManager | None = None

async def main():
    global LIVE, TM, COALESCE
    await head_line(f"[INFO] Лог файл: {LOG_PATH}")

    # S1
    if CAPTURE_S1:
        if not S1_DEVICE:
            await head_line("[WARN] S1_DEVICE не найден. Отключаю S1.")
            cap_s1 = False
        else:
            cap_s1 = True
            await head_line(f"[INFO] S1_DEVICE = {S1_DEVICE}")
    else:
        cap_s1 = False
        await head_line("[INFO] S1 отключён (CAPTURE_S1=0).")

    # S2
    if CAPTURE_S2:
        await head_line(f"[INFO] S2_TARGET (режим/имя) = {S2_TARGET or 'AUTO_DEFAULT_SINK'}")
    else:
        await head_line("[INFO] S2 отключён (CAPTURE_S2=0).")

    await head_line("[INFO] В звонке выберите маршрут Bluetooth на телефоне: jupiter-Asp")

    # UI и менеджеры
    LIVE = LiveManager(use_rich=True);   await LIVE.start()
    TM   = TranslationManager(LIVE);     await TM.start()
    COALESCE = Coalescer(LIVE, TM);      await COALESCE.start()

    # 1) старт RT-сессий (с попыткой перевода при флаге)
    sessions = []
    if CAPTURE_S2:
        sessions.append(("S2",) + (await start_sm("S2", ENABLE_TRANSLATION)))
    if cap_s1:
        sessions.append(("S1",) + (await start_sm("S1", ENABLE_TRANSLATION)))
    if not sessions:
        await head_line("[ERROR] Нет активных сессий (оба канала отключены). Включите CAPTURE_S1 и/или CAPTURE_S2.")
        await COALESCE.stop(); await TM.stop(); await LIVE.stop()
        return

    reader_tasks = [asyncio.create_task(s[2]()) for s in sessions]

    # 2) ждём старта/фоллбэк без перевода при необходимости (и с паузой для освобождения квоты)
    active = []
    for i, s in enumerate(sessions):
        name, ws, r, send, fin, started, err = s
        st, info = await _wait_start_or_error(name, started, err, timeout=8.0)
        if st == "started":
            active.append(s); continue

        do_retry = False
        if st == "error":
            et, rsn = info
            await head_line(f"[{name}] Старт не удался: {et} — {rsn}")
            if et == "protocol_error" and _is_translation_schema_error(et, rsn):
                do_retry = True
        elif st == "timeout" and ENABLE_TRANSLATION:
            await head_line(f"[{name}] Timeout ожидания RecognitionStarted — пробую без перевода…")
            do_retry = True

        if do_retry:
            await _close_reader_and_ws(s, reader_tasks[i])
            await asyncio.sleep(0.8)  # дать квоте освободиться
            ws2, r2, send2, fin2, started2, err2 = await start_sm(name, False)
            sessions[i] = (name, ws2, r2, send2, fin2, started2, err2)
            reader_tasks[i] = asyncio.create_task(r2())
            st2, info2 = await _wait_start_or_error(name, started2, err2, timeout=8.0)
            if st2 == "started":
                await head_line(f"[FALLBACK] {name}: перевода нет, работаем без него.")
                active.append(sessions[i])
            else:
                await head_line(f"[ERROR] {name}: повторный старт без перевода не удался ({st2}: {info2}).")
        # если не do_retry — канал считаем упавшим

    if not active:
        await head_line("[ERROR] Ни одна RT-сессия не запустилась. Проверьте ключ/квоты/сеть и перезапустите.")
        for t in reader_tasks:
            with contextlib.suppress(Exception): t.cancel()
        await COALESCE.stop(); await TM.stop(); await LIVE.stop()
        return

    # 3) квоты (на случай async-ошибок после старта)
    still = []
    for s in active:
        name, ws, r, send, fin, started, err = s
        if (err.get("type") or "").lower() == "quota_exceeded":
            await head_line(f"[FALLBACK] {name} отклонён квотой — отключаю.")
            with contextlib.suppress(Exception): await fin()
        else:
            still.append(s)
    active = still
    if not active:
        await head_line("[ERROR] quota_exceeded для всех сессий. Закройте висящие RT-сессии и перезапустите.")
        for t in reader_tasks:
            with contextlib.suppress(Exception): t.cancel()
        await COALESCE.stop(); await TM.stop(); await LIVE.stop()
        return

    # 4) выбираем sink для S2
    tasks = []
    have_S2 = any(s[0]=="S2" for s in active)
    actual_s2 = None
    if have_S2:
        mode = S2_TARGET or "AUTO_DEFAULT_SINK"
        if mode in ("AUTO_DEFAULT_SINK","AUTO_BT_SINK","AUTO_ALSA_SINK"):
            await info_line(f"[S2] режим {mode} — выбираю sink…")
        else:
            await info_line(f"[S2] ожидаемый sink: {mode}")
        actual_s2 = await wait_for_sink(mode, timeout_sec=180)
        if not actual_s2:
            await head_line("[ERROR] Не удалось найти подходящий sink для S2.")
            for s in active:
                with contextlib.suppress(Exception): await s[4]()
            for t in reader_tasks:
                with contextlib.suppress(Exception): t.cancel()
            await COALESCE.stop(); await TM.stop(); await LIVE.stop()
            return
        await info_line(f"[S2] выбран sink: {actual_s2}")

    # 5) старт захвата
    for s in active:
        name, ws, r, send, fin, started, err = s
        if name == "S2":
            cmd = ["pw-record","--target",actual_s2,"--properties","stream.capture.sink=true",
                   "--format","s16","--channels","1","--rate",str(SAMPLE_RATE),"-"]
            tasks.append(asyncio.create_task(run_pw_record(cmd, send, "S2_capture")))
        elif name == "S1":
            cmd = ["pw-record","--target",S1_DEVICE,"--format","s16","--channels","1","--rate",str(SAMPLE_RATE),"-"]
            tasks.append(asyncio.create_task(run_pw_record(cmd, send, "S1_capture")))

    # 6) завершение по Ctrl+C
    stop = asyncio.Event()
    def _sig(*_): stop.set()
    for sgn in (signal.SIGINT, signal.SIGTERM): signal.signal(sgn, _sig)
    await stop.wait()

    # 7) закрытие
    for s in active:
        with contextlib.suppress(Exception): await s[4]()  # fin
    for t in reader_tasks + tasks:
        with contextlib.suppress(Exception): t.cancel()
    # дать сетевым транспортам корректно схлопнуться
    await asyncio.sleep(0.2)
    await COALESCE.stop(); await TM.stop(); await LIVE.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, asyncio, json, signal, datetime as dt, subprocess, contextlib, time, re as _re
import websockets
import urllib.request, urllib.error

# ===================== ТЕСТОВЫЕ ПЕРЕМЕННЫЕ (цель/контекст) =====================
DIALOG_GOAL_EN_DEFAULT    = "I'm calling my electricity provider to arrange and order a smart meter as soon as possible."
DIALOG_CONTEXT_EN_DEFAULT = "United Kingdom context, I am a 35-year-old male customer. Be concise, polite and goal-driven."

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

# ---------- Отладка/вывод ----------
DEBUG  = ENV.get("DEBUG", "0").lower() in ("1","true","yes")

# ---------- Подсказки (локальные модели через reply_engine) ----------
ENABLE_SUGGEST       = ENV.get("ENABLE_SUGGEST", "1").lower() in ("1","true","yes")
SUGGEST_MAX_HISTORY  = int(ENV.get("SUGGEST_MAX_HISTORY", "10"))  # для совместимости со стеком истории
SUGGEST_TIMEOUT      = float(ENV.get("SUGGEST_TIMEOUT", "6.0"))   # не используется в локальном режиме
SUGGEST_TEMPERATURE  = float(ENV.get("SUGGEST_TEMPERATURE", "0.2"))  # не используется
SUGGEST_MODEL        = ENV.get("SUGGEST_MODEL", "gpt-4.1-nano")   # используется только в RU->EN переводе цели

# ---------- OpenAI (только для автоперевода цели RU->EN при наличии ключа) ----------
OPENAI_API_KEY       = (ENV.get("OPENAI_API") or "").strip()

# ---- Маршрутизация/захват S2 ----
REQUIRE_BT_FOR_S2       = ENV.get("REQUIRE_BT_FOR_S2", "0").lower() in ("1","true","yes")
ALLOW_S2_ALSA_FALLBACK  = ENV.get("ALLOW_S2_ALSA_FALLBACK", "1").lower() in ("1","true","yes")
S2_WAIT_FOR_APPEAR_SEC  = float(ENV.get("S2_WAIT_FOR_APPEAR_SEC", "15.0"))
S2_APPEAR_POLL_MS       = int(ENV.get("S2_APPEAR_POLL_MS", "300"))

# ---- Подавление эха ----
ECHO_SIM_THRESHOLD  = float(ENV.get("ECHO_SIM_THRESHOLD", "0.75"))
ECHO_WINDOW_SEC     = float(ENV.get("ECHO_WINDOW_SEC", "1.2"))
MAX_S1_CACHE        = int(ENV.get("MAX_S1_CACHE", "3"))

# ---- Speechmatics ----
SM_ENDPOINT = ENV.get("SPEECHMATICS_WSS", "wss://eu2.rt.speechmatics.com/v2/")
SAMPLE_RATE = int(ENV.get("SAMPLE_RATE", "16000"))
ENCODING = "pcm_s16le"
MAX_DELAY = float(ENV.get("MAX_DELAY", "1.3"))
MAX_DELAY_MODE = ENV.get("MAX_DELAY_MODE", "flexible")
EOU_SILENCE = float(ENV.get("EOU_SILENCE", "0.8"))

# ---- live-троттлинг/эвристики ----
PARTIAL_DEBOUNCE_MS = int(ENV.get("PARTIAL_DEBOUNCE_MS", "350"))
MIN_DELTA_CHARS     = int(ENV.get("MIN_DELTA_CHARS", "8"))
MIN_WORDS_PARTIAL   = int(ENV.get("MIN_WORDS_PARTIAL", "2"))
SILENCE_FLUSH_MS    = int(ENV.get("SILENCE_FLUSH_MS", "900"))
EOU_FINALIZE_MS     = int(ENV.get("EOU_FINALIZE_MS", "1100"))

# ---- перевод (только финалы) ----
ENABLE_TRANSLATION     = ENV.get("ENABLE_TRANSLATION", "0").lower() in ("1","true","yes")
TRANSLATE_TO           = (ENV.get("TRANSLATE_TO", "ru") or "ru").strip().lower()
TRANSLATION_WINDOW_MS  = int(ENV.get("TRANSLATION_WINDOW_MS", "2000"))

PRINT_AUDIOADDED    = ENV.get("PRINT_AUDIOADDED", "0").lower() in ("1","true","yes")

# ---- управление захватом линий ----
CAPTURE_S1 = ENV.get("CAPTURE_S1", "1").lower() in ("1","true","yes")
CAPTURE_S2 = ENV.get("CAPTURE_S2", "1").lower() in ("1","true","yes")

# ---- логи ----
LOG_DIR = os.path.expanduser("call_logs"); os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"call_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
def ts(): return dt.datetime.now().strftime("%H:%M:%S")

# ------------- цвета / вывод -------------
def _supports_color():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
USE_COLOR = _supports_color()
RESET = "\033[0m" if USE_COLOR else ""
DIM   = "\033[2m" if USE_COLOR else ""
RED   = "\033[31m" if USE_COLOR else ""
GRN   = "\033[32m" if USE_COLOR else ""
BLU   = "\033[34m" if USE_COLOR else ""
CYAN  = "\033[36m" if USE_COLOR else ""
ITAL  = "\033[3m"  if USE_COLOR else ""
ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(s: str) -> str: return ANSI_RE.sub("", s)
_print_lock = asyncio.Lock()

# глобальный счётчик «сноски» для реплик
MSG_ID = 0
def next_msg_id() -> int:
    global MSG_ID; MSG_ID += 1; return MSG_ID

async def print_and_log(line: str):
    async with _print_lock:
        print(line, flush=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(strip_ansi(line) + "\n")

async def info_line(s: str):
    if DEBUG:
        await print_and_log(f"{DIM}{s}{RESET}" if USE_COLOR else s)

async def head_line(s: str):
    await print_and_log(f"{CYAN}{s}{RESET}" if USE_COLOR else s)

# ------------------- Live UI -------------------
HAVE_RICH = True
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except Exception:
    HAVE_RICH = False

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

    async def print_final_with_extras(self, label: str, text: str, tr_text: str | None = None) -> int:
        msg_id = next_msg_id()
        t0 = ts()
        label_col = RED if label == "S1" else GRN
        await print_and_log(f"{t0} | [#{msg_id}] {label_col}{label}{RESET}: {text}")
        if ENABLE_TRANSLATION:
            if tr_text and tr_text.strip():
                await print_and_log(f"{DIM}{t0} | [#{msg_id}] {label} → RU: {tr_text.strip()}{RESET}")
            else:
                await print_and_log(f"{DIM}{t0} | [#{msg_id}] {label} → RU: —{RESET}")
        # ВАЖНО: НЕ печатаем placeholder '↳ SUGGEST[...] —'
        return msg_id

# ------------------- менеджер перевода -------------------
class TranslationManager:
    def __init__(self, live: LiveManager):
        self.live = live
        self._parts = {"S1": [], "S2": []}
        self._pending = {"S1": None, "S2": None}
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

# ------------------- Менеджер подсказок (локальные модели через reply_engine) -------------------
from reply_engine import ReplyEngine  # модуль из файла reply_engine.py

class SuggestionManager:
    """
    Адаптер над ReplyEngine.
    - Во время партиалов S2 даёт быстрые подсказки (llama3.2:1b).
    - На финале S2 — краткая финальная (та же модель).
    """
    def __init__(self, goal_en: str, context_en: str):
        self.enabled = ENABLE_SUGGEST
        self.engine = ReplyEngine(goal=goal_en, context=context_en)
        self.queue = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        # анти-спам для вывода (доп. к внутреннему троттлингу движка)
        self._last_fast_text = ""
        self._last_fast_ts_ms = 0.0
        self._fast_dedupe_window_ms = 8000.0  # не печатать одинаковое в течение 8с

    def add_history(self, label: str, text: str):
        self.engine.add_history(label, text)

    async def start(self):
        if not self.enabled:
            await head_line("[SUGGEST] Подсказки отключены (ENABLE_SUGGEST=0).")
            return
        self.worker_task = asyncio.create_task(self._worker())
        await head_line("[SUGGEST] ReplyEngine инициализирован.")

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            with contextlib.suppress(Exception):
                await self.worker_task

    async def schedule_fast_partial(self, s2_partial_text: str):
        if not self.enabled:
            return
        await self.queue.put({"kind": "fast", "partial": s2_partial_text})

    async def schedule_for_s2(self, anchor_id: int):
        if not self.enabled:
            return
        await self.queue.put({"kind": "final", "anchor": anchor_id})

    async def _worker(self):
        while True:
            job = await self.queue.get()
            try:
                if job.get("kind") == "fast":
                    text = job.get("partial") or ""
                    reply = await self.engine.fast_reply_from_partial(text)
                    if reply:
                        # дедупликация и минимальный интервал вывода
                        now_ms = time.monotonic() * 1000.0
                        if (reply == self._last_fast_text) and ((now_ms - self._last_fast_ts_ms) < self._fast_dedupe_window_ms):
                            pass
                        else:
                            await print_and_log(f"{BLU}{ts()} |   ↳ FAST: {reply}{RESET}")
                            self._last_fast_text = reply
                            self._last_fast_ts_ms = now_ms
                elif job.get("kind") == "final":
                    anchor = job.get("anchor")
                    reply = await self.engine.final_reply()
                    if reply:
                        await print_and_log(f"{BLU}{ts()} |   ↳ SUGGEST[#{anchor}]: {reply}{RESET}")
            except Exception as e:
                await info_line(f"[SUGGEST] worker error: {e}")
            finally:
                self.queue.task_done()

# ------------------- PipeWire helpers -------------------
def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8","ignore")
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8","ignore")
    except Exception:
        return ""

_PREFIX_RE = _re.compile(r'\b((?:bluez|alsa)_(?:input|output|source)\.[A-Za-z0-9_.:-]+)\b')

def list_node_names_wpctl():
    out = run_cmd(["wpctl","status","--name"])
    return set(_PREFIX_RE.findall(out))

def list_node_names_pwdump():
    out = run_cmd(["pw-dump"])
    names = set()
    for m in _re.finditer(r'"node\.name"\s*:\s*"([^"]+)"', out):
        n = m.group(1)
        if n.startswith(("bluez_input.","bluez_source.","bluez_output.","alsa_input.","alsa_output.")):
            names.add(n)
    return names

def list_node_names():
    names = set()
    names |= list_node_names_wpctl()
    if not names:
        names |= list_node_names_pwdump()
    return sorted(names)

def node_exists(name: str) -> bool:
    return name in list_node_names()

def list_bluez_nodes():
    return [n for n in list_node_names() if n.startswith(("bluez_output.","bluez_input.","bluez_source."))]

def is_sink_node(name: str) -> bool:
    return name.startswith(("alsa_output.","bluez_output."))

def is_source_node(name: str) -> bool:
    return name.startswith(("alsa_input.","bluez_input.","bluez_source."))

def list_bluez_inputs():
    return [n for n in list_node_names() if n.startswith(("bluez_input.","bluez_source."))]

def list_bluez_sinks():
    return [n for n in list_node_names() if n.startswith("bluez_output.")]

def get_default_sink_name():
    out = run_cmd(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"])
    m = _re.search(r'node\.name\s*=\s*"([^"]+)"', out)
    return m.group(1) if m else ""

def find_mic_source_prefer_usb():
    for n in list_node_names():
        if n.startswith("alsa_input.") and any(k in n for k in ("HyperX","DuoCast","usb-","Trust_PC_Headset","Generalplus")):
            return n
    for n in list_node_names():
        if n.startswith("alsa_input."): return n
    return ""

async def wait_for_exact_node(target_name: str, timeout_sec: float) -> bool:
    t0 = time.monotonic()
    while (time.monotonic() - t0) < timeout_sec:
        if node_exists(target_name):
            return True
        await asyncio.sleep(S2_APPEAR_POLL_MS/1000.0)
    return node_exists(target_name)

async def wait_for_any_bluez_input(timeout_sec: float) -> list[str]:
    t0 = time.monotonic()
    while (time.monotonic() - t0) < timeout_sec:
        ins = list_bluez_inputs()
        if ins:
            return ins
        await asyncio.sleep(S2_APPEAR_POLL_MS/1000.0)
    return list_bluez_inputs()

def build_s2_candidates(primary_mode: str) -> list[str]:
    names = list_node_names()
    bt_inputs = [n for n in names if n.startswith(("bluez_input.","bluez_source."))]
    bt_sinks  = [n for n in names if n.startswith("bluez_output.")]
    alsa_snks = [n for n in names if n.startswith("alsa_output.")]
    def_snk   = get_default_sink_name()

    if primary_mode and primary_mode not in ("AUTO_BT_SINK","AUTO_BT_AUTO","AUTO_DEFAULT_SINK","AUTO_ALSA_SINK"):
        return [primary_mode] if primary_mode in names else []

    cands: list[str] = []
    cands.extend(bt_inputs)   # 1) HFP/HSP sources
    cands.extend(bt_sinks)    # 2) A2DP sinks (monitor)
    if ALLOW_S2_ALSA_FALLBACK and (not REQUIRE_BT_FOR_S2 or not (bt_inputs or bt_sinks)):
        if def_snk:
            cands.append(def_snk)
        for n in alsa_snks:
            if n != def_snk:
                cands.append(n)

    seen = set(); uniq = []
    for n in cands:
        if n not in seen:
            uniq.append(n); seen.add(n)
    return uniq

# --- Инициализация устройств S1/S2 ---
S1_DEVICE = ENV.get("S1_DEVICE") or find_mic_source_prefer_usb()
S2_TARGET = ENV.get("S2_TARGET") or "AUTO_BT_AUTO"

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

# ------------------- S2 capture controller & routing guard -------------------
class S2CaptureController:
    def __init__(self, send_func, sample_rate: int):
        self.send_func = send_func
        self.sample_rate = sample_rate
        self.current_node = None
        self.task: asyncio.Task | None = None

    async def start(self, node_name: str):
        self.current_node = node_name
        if is_sink_node(node_name):
            await head_line(f"[S2_CAPTURE] start (monitor) → {node_name}")
            cmd = [
                "pw-record",
                "--monitor",
                "--target", node_name,
                "--format", "s16", "--channels", "1", "--rate", str(self.sample_rate),
                "-"
            ]
        else:
            await head_line(f"[S2_CAPTURE] start (direct) → {node_name}")
            cmd = [
                "pw-record",
                "--target", node_name,
                "--format", "s16", "--channels", "1", "--rate", str(self.sample_rate),
                "-"
            ]
        self.task = asyncio.create_task(run_pw_record(cmd, self.send_func, "S2_capture"))

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(Exception):
                await self.task
            self.task = None
        self.current_node = None

    async def switch_to(self, node_name: str):
        if node_name == self.current_node:
            return
        await head_line(f"[S2_CAPTURE] switch ({'monitor' if is_sink_node(node_name) else 'direct'}) → {node_name}")
        await self.stop()
        await self.start(node_name)

class RoutingGuard:
    def __init__(self, s2ctrl: S2CaptureController, candidates: list[str]):
        self.s2ctrl = s2ctrl
        self.candidates = candidates[:]
        self.idx = 0
        self.task: asyncio.Task | None = None
        self.last = {"S1": {"txt":"", "t":0.0}, "S2": {"txt":"", "t":0.0}}
        self._last_switch_t = 0.0
        self.s1_cache: list[tuple[str,float]] = []  # (text, t)

    async def start(self):
        self.task = asyncio.create_task(self._monitor())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(Exception):
                await self.task

    async def note(self, label: str, txt: str):
        now = time.monotonic()
        self.last[label]["txt"] = txt or ""
        self.last[label]["t"] = now
        if label == "S1":
            self.s1_cache.append((txt or "", now))
            if len(self.s1_cache) > MAX_S1_CACHE:
                self.s1_cache = self.s1_cache[-MAX_S1_CACHE:]

    def _sim(self, a: str, b: str) -> float:
        def _nw(s):
            s = s.lower()
            s = _re.sub(r"[^a-z0-9' ]+", " ", s)
            return [w for w in s.split() if w]
        A, B = set(_nw(a)), set(_nw(b))
        if not A and not B: return 1.0
        if not A or not B:  return 0.0
        inter = len(A & B); union = max(1, len(A | B))
        return inter / union

    def is_echo_of_recent_s1(self, s2_txt: str) -> bool:
        now = time.monotonic()
        if len(s2_txt.strip()) <= 3:
            return True
        for (t1, t1_time) in list(reversed(self.s1_cache)):
            if (now - t1_time) > ECHO_WINDOW_SEC:
                break
            if self._sim(s2_txt, t1) >= ECHO_SIM_THRESHOLD:
                return True
        return False

    async def maybe_switch_due_to_echo(self):
        now = time.monotonic()
        if len(self.candidates) <= 1:
            return
        if (now - self._last_switch_t) < 5.0:
            return
        new_idx = (self.idx + 1) % len(self.candidates)
        await head_line(f"[ROUTE] Обнаружено эхо-дублирование S2 → пробую {self.candidates[new_idx]}")
        self.idx = new_idx
        await self.s2ctrl.switch_to(self.candidates[self.idx])
        self._last_switch_t = now

    async def _monitor(self):
        try:
            while True:
                await asyncio.sleep(0.6)
                now = time.monotonic()
                s1_recent = (now - self.last["S1"]["t"]) < 3.0
                s2_silence = (now - self.last["S2"]["t"]) > 4.0
                if s1_recent and s2_silence and len(self.candidates) > 1:
                    if (now - self._last_switch_t) >= 6.0:
                        new_idx = (self.idx + 1) % len(self.candidates)
                        await head_line(f"[ROUTE] Нет S2-потока → пробую {self.candidates[new_idx]}")
                        self.idx = new_idx
                        await self.s2ctrl.switch_to(self.candidates[self.idx])
                        self._last_switch_t = now
        except asyncio.CancelledError:
            pass

# ------------------- Коалесцер реплик (FAST на партиалах S2) -------------------
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
        joined_now = self._joined(label)
        await self.live.show_live_text(label, joined_now)

        if label == "S2" and 'SM' in globals() and SM:
            await SM.schedule_fast_partial(joined_now)

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

        if 'ROUTE_GUARD' in globals() and ROUTE_GUARD:
            await ROUTE_GUARD.note(label, txt)

        if label == "S2" and 'ROUTE_GUARD' in globals() and ROUTE_GUARD:
            try:
                if ROUTE_GUARD.is_echo_of_recent_s1(txt):
                    await info_line("[ECHO] Подавил S2 как эхо собственной реплики")
                    await ROUTE_GUARD.maybe_switch_due_to_echo()
                    self._parts[label].clear()
                    self._partial[label] = ""
                    self._last_activity[label] = 0.0
                    return
            except Exception:
                pass

        tr_now = None
        if TM:
            tr_now = await TM.on_utterance_finalized(label)

        await self.live.clear_live_line(label)
        msg_id = await self.live.print_final_with_extras(label, txt, tr_text=tr_now)

        if 'SM' in globals() and SM:
            SM.add_history(label, txt)
            if ENABLE_SUGGEST and label == "S2":
                await SM.schedule_for_s2(anchor_id=msg_id)

        self._parts[label].clear()
        self._partial[label] = ""
        self._last_activity[label] = 0.0

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
    with contextlib.suppress(Exception):
        await sess[4]()  # fin()
    with contextlib.suppress(Exception):
        reader_task.cancel()
        await asyncio.sleep(0)

# --- RU -> EN автоперевод цели (опционально, при наличии OPENAI_API) ---
def _looks_russian(s: str) -> bool:
    return bool(_re.search(r"[А-Яа-яЁё]", s or ""))

def _translate_ru_goal_to_en(text: str) -> str:
    key = OPENAI_API_KEY
    if not key:
        return text
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": SUGGEST_MODEL or "gpt-4.1-nano",
                "messages": [
                    {
                        "role": "system",
                        "content": "Translate the user's goal into concise, natural UK English. Return only the translation text without quotes."
                    },
                    {"role": "user", "content": text}
                ],
                "temperature": 0,
                "max_tokens": 120
            }).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        out = out.strip()
        if out:
            out = _re.sub(r'^\s*[\"“](.*?)[\"”]\s*$', r'\1', out)
            print(f'Goal:\n{out}')
            return out
    except Exception as e:
        if DEBUG:
            print(f"[GOAL] RU->EN translation failed: {e}", file=sys.stderr)
    return text

_raw_goal = (ENV.get("DIALOG_GOAL_EN") or DIALOG_GOAL_EN_DEFAULT).strip()
DIALOG_GOAL_EN = _translate_ru_goal_to_en(_raw_goal) if _looks_russian(_raw_goal) else _raw_goal
DIALOG_CONTEXT_EN = (ENV.get("DIALOG_CONTEXT_EN") or DIALOG_CONTEXT_EN_DEFAULT).strip()

# ------------------- main -------------------
COALESCE: Coalescer | None = None
LIVE: LiveManager | None = None
TM: TranslationManager | None = None
SM: SuggestionManager | None = None
ROUTE_GUARD: RoutingGuard | None = None

async def main():
    global LIVE, TM, COALESCE, SM, ROUTE_GUARD
    await head_line(f"[INFO] Лог файл: {LOG_PATH}")

    # S1 — только источник (микрофон)
    if CAPTURE_S1:
        if not S1_DEVICE:
            await head_line("[WARN] S1_DEVICE не найден. Отключаю S1.")
            cap_s1 = False
        elif not S1_DEVICE.startswith("alsa_input."):
            await head_line(f"[ERROR] S1_DEVICE='{S1_DEVICE}' не является источником (alsa_input.*). Отключаю S1.")
            cap_s1 = False
        else:
            cap_s1 = True
            await head_line(f"[INFO] S1_DEVICE = {S1_DEVICE}")
    else:
        cap_s1 = False
        await head_line("[INFO] S1 отключён (CAPTURE_S1=0).")

    # S2 — входящий поток с телефона (HFP/source предпочтителен)
    if CAPTURE_S2:
        await head_line(f"[INFO] S2_TARGET (режим/имя) = {S2_TARGET or 'AUTO_BT_AUTO'}")
    else:
        await head_line("[INFO] S2 отключён (CAPTURE_S2=0).")

    await head_line("[INFO] В звонке выберите маршрут Bluetooth на телефоне: jupiter-Asp")

    # UI и менеджеры
    LIVE = LiveManager(use_rich=True);   await LIVE.start()
    TM   = TranslationManager(LIVE);     await TM.start()

    SM   = SuggestionManager(goal_en=DIALOG_GOAL_EN, context_en=DIALOG_CONTEXT_EN)
    await SM.start()

    COALESCE = Coalescer(LIVE, TM);      await COALESCE.start()

    # 1) старт RT-сессий
    sessions = []
    if CAPTURE_S2:
        sessions.append(("S2",) + (await start_sm("S2", ENABLE_TRANSLATION)))
    if cap_s1:
        sessions.append(("S1",) + (await start_sm("S1", ENABLE_TRANSLATION)))
    if not sessions:
        await head_line("[ERROR] Нет активных сессий (оба канала отключены).")
        await COALESCE.stop(); await TM.stop(); await SM.stop(); await LIVE.stop()
        return

    reader_tasks = [asyncio.create_task(s[2]()) for s in sessions]

    # 2) ждём старта/фоллбэка без перевода
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
            await head_line(f"[{name}] Timeout RecognitionStarted — пробую без перевода…")
            do_retry = True

        if do_retry:
            await _close_reader_and_ws(s, reader_tasks[i])
            await asyncio.sleep(0.8)
            ws2, r2, send2, fin2, started2, err2 = await start_sm(name, False)
            sessions[i] = (name, ws2, r2, send2, fin2, started2, err2)
            reader_tasks[i] = asyncio.create_task(r2())
            st2, info2 = await _wait_start_or_error(name, started2, err2, timeout=8.0)
            if st2 == "started":
                await head_line(f"[FALLBACK] {name}: перевода нет, работаем без него.")
                active.append(sessions[i])
            else:
                await head_line(f"[ERROR] {name}: повторный старт без перевода не удался ({st2}: {info2}).")

    if not active:
        await head_line("[ERROR] Ни одна RT-сессия не запустилась.")
        for t in reader_tasks:
            with contextlib.suppress(Exception): t.cancel()
        await COALESCE.stop(); await TM.stop(); await SM.stop(); await LIVE.stop()
        return

    # 3) квоты
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
        await head_line("[ERROR] quota_exceeded для всех сессий.")
        for t in reader_tasks:
            with contextlib.suppress(Exception): t.cancel()
        await COALESCE.stop(); await TM.stop(); await SM.stop(); await LIVE.stop()
        return

    # 4) кандидаты для S2 и старт захвата + роутер
    tasks = []
    have_S2 = any(s[0] == "S2" for s in active)
    if have_S2:
        s2_session = [s for s in active if s[0] == "S2"][0]
        name, ws, r, send, fin, started, err = s2_session

        explicit = (S2_TARGET and S2_TARGET not in ("AUTO_BT_SINK","AUTO_BT_AUTO","AUTO_DEFAULT_SINK","AUTO_ALSA_SINK"))
        if explicit and not node_exists(S2_TARGET):
            await head_line(f"[S2] Ожидаю появление узла {S2_TARGET} (до {S2_WAIT_FOR_APPEAR_SEC:.0f} с). "
                            f"Во время звонка выберите на телефоне аудио 'Bluetooth-гарнитура/jupiter-Asp'.")
            ok = await wait_for_exact_node(S2_TARGET, S2_WAIT_FOR_APPEAR_SEC)
            if not ok:
                await head_line(f"[ERROR] Узел {S2_TARGET} так и не появился — S2 будет отключён.")
                with contextlib.suppress(Exception): await fin()
                active = [s for s in active if s[0] != "S2"]
                have_S2 = False

        candidates = []
        if have_S2:
            if explicit:
                candidates = [S2_TARGET]
            else:
                candidates = build_s2_candidates(S2_TARGET or "AUTO_BT_AUTO")
                if REQUIRE_BT_FOR_S2 and not any(c.startswith(("bluez_input.","bluez_source.")) for c in candidates):
                    await head_line(f"[S2] Жду появления HFP (bluez_input/source) до {S2_WAIT_FOR_APPEAR_SEC:.0f} с…")
                    ins = await wait_for_any_bluez_input(S2_WAIT_FOR_APPEAR_SEC)
                    if ins:
                        candidates = build_s2_candidates(S2_TARGET or "AUTO_BT_AUTO")
                    else:
                        if ALLOW_S2_ALSA_FALLBACK:
                            await head_line("[WARN] HFP не появился — разрешён ALSA-фоллбэк, использую monitor дефолтного sink.")
                            candidates = build_s2_candidates("AUTO_DEFAULT_SINK")
                        else:
                            await head_line("[ERROR] HFP не появился, фоллбэк запрещён (ALLOW_S2_ALSA_FALLBACK=0). Отключаю S2.")
                            with contextlib.suppress(Exception): await fin()
                            active = [s for s in active if s[0] != "S2"]
                            have_S2 = False

        if have_S2 and not candidates:
            await head_line("[ERROR] Нет подходящих нод для S2.")
            with contextlib.suppress(Exception): await fin()
            active = [s for s in active if s[0] != "S2"]
            have_S2 = False

        if have_S2:
            s2ctrl = S2CaptureController(send, SAMPLE_RATE)
            await s2ctrl.start(candidates[0])

            global ROUTE_GUARD
            ROUTE_GUARD = RoutingGuard(s2ctrl, candidates)
            await ROUTE_GUARD.start()
            tasks.append(s2ctrl.task)

    # 5) старт захвата S1 (строго источник)
    for s in active:
        name, ws, r, send, fin, started, err = s
        if name == "S1":
            if not S1_DEVICE.startswith("alsa_input."):
                await head_line(f"[ERROR] Пропуск S1: {S1_DEVICE} не является источником (alsa_input.*)")
                continue
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
    if ROUTE_GUARD:
        await ROUTE_GUARD.stop()
    await asyncio.sleep(0.2)
    await COALESCE.stop(); await TM.stop(); await SM.stop(); await LIVE.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, asyncio, json, signal, datetime as dt, subprocess, contextlib, time, re as _re
import websockets

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
ENABLE_TRANSLATION = ENV.get("ENABLE_TRANSLATION", "0").lower() in ("1","true","yes")
SM_ENDPOINT = ENV.get("SPEECHMATICS_WSS", "wss://eu2.rt.speechmatics.com/v2/")

SAMPLE_RATE = int(ENV.get("SAMPLE_RATE", "16000"))
ENCODING = "pcm_s16le"
MAX_DELAY = float(ENV.get("MAX_DELAY", "1.3"))
MAX_DELAY_MODE = ENV.get("MAX_DELAY_MODE", "flexible")
EOU_SILENCE = float(ENV.get("EOU_SILENCE", "0.8"))

# анти-спам партиалов
PARTIAL_DEBOUNCE_MS = int(ENV.get("PARTIAL_DEBOUNCE_MS", "700"))
MIN_DELTA_CHARS     = int(ENV.get("MIN_DELTA_CHARS", "8"))
MIN_WORDS_PARTIAL   = int(ENV.get("MIN_WORDS_PARTIAL", "2"))
PRINT_AUDIOADDED    = ENV.get("PRINT_AUDIOADDED", "0").lower() in ("1","true","yes")

# управление захватом линий
CAPTURE_S1 = ENV.get("CAPTURE_S1", "1").lower() in ("1","true","yes")
CAPTURE_S2 = ENV.get("CAPTURE_S2", "1").lower() in ("1","true","yes")

# логи
LOG_DIR = os.path.expanduser("call_logs"); os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"call_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
def ts(): return dt.datetime.now().strftime("%H:%M:%S")

# ------------------- вывод (цвета) -------------------
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

def _fmt_label(label: str) -> str:
    if label == "S1": return f"{RED}S1{RESET}" if USE_COLOR else "S1"
    if label == "S2": return f"{GRN}S2{RESET}" if USE_COLOR else "S2"
    return label

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

async def say_line(label: str, text: str, partial: bool):
    tag = _fmt_label(label)
    suf = f" {ITAL}…(partial){RESET}" if (partial and USE_COLOR) else (" …(partial)" if partial else "")
    await print_and_log(f"{ts()} | {tag}: {text}{suf}")

async def say_tr_line(label: str, text: str, partial: bool):
    tag = _fmt_label(label)
    suf = f" {ITAL}…(partial){RESET}" if (partial and USE_COLOR) else (" …(partial)" if partial else "")
    await print_and_log(f"{ts()} | {tag} → RU: {text}{suf}")

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

def find_first_bluez_output():
    for n in list_node_names():
        if n.startswith("bluez_output."): return n
    return ""

def find_bt_sink_any():
    # первый доступный bluetooth sink
    names = list_node_names()
    for n in names:
        if n.startswith("bluez_output."):
            return n
    return ""

def find_alsa_sink_any():
    # первый доступный alsa sink (проводной/USB/HDMI)
    names = list_node_names()
    for n in names:
        if n.startswith("alsa_output."):
            return n
    return ""

def get_default_sink_name():
    # берём имя дефолтного sink через inspect
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
    """
    Ожидаем появление нужного sink:
      - "AUTO_DEFAULT_SINK" : текущий дефолт (@DEFAULT_AUDIO_SINK@)
      - "AUTO_BT_SINK"      : первый bluez_output.*
      - "AUTO_ALSA_SINK"    : первый alsa_output.*
      - <конкретное_имя>    : точное совпадение
    Возвращает имя sink или "" при таймауте.
    """
    t0 = dt.datetime.now()

    def pick():
        if mode_or_name == "AUTO_DEFAULT_SINK":
            return get_default_sink_name() or ""
        if mode_or_name == "AUTO_BT_SINK":
            return find_bt_sink_any()
        if mode_or_name == "AUTO_ALSA_SINK":
            return find_alsa_sink_any()
        # точное имя
        names = list_node_names()
        return mode_or_name if mode_or_name and mode_or_name in names else ""

    # мгновенная попытка
    n0 = pick()
    if n0:
        return n0

    # ожидание
    while True:
        cand = pick()
        if cand:
            return cand
        if (dt.datetime.now() - t0).total_seconds() > timeout_sec:
            return ""
        await asyncio.sleep(0.3)

# --- Инициализация устройств S1/S2 (S1 сразу, S2 выбираем позже по режиму) ---
S1_DEVICE = ENV.get("S1_DEVICE") or find_hyperx_mic()
S2_TARGET = ENV.get("S2_TARGET") or ""  # может быть именем или AUTO_* режимом

# ------------------- частичная печать: состояние -------------------
_last_partial_text = {"S1":"","S2":"","S1→RU":"","S2→RU":""}
_last_partial_time = {"S1":0.0,"S2":0.0,"S1→RU":0.0,"S2→RU":0.0}
_punct_re = _re.compile(r"[.!?…]$")

def _word_count(s: str) -> int:
    return len([w for w in s.strip().split() if w])

def _should_print_partial(label: str, txt: str) -> bool:
    now = time.monotonic() * 1000.0
    last_txt = _last_partial_text.get(label, "")
    last_ts  = _last_partial_time.get(label, 0.0)

    if not txt: return False
    grew = (len(txt) - len(last_txt)) >= MIN_DELTA_CHARS
    enough_words = _word_count(txt) >= MIN_WORDS_PARTIAL
    punct = bool(_punct_re.search(txt))
    debounce_ok = (now - last_ts) >= PARTIAL_DEBOUNCE_MS

    if (grew and enough_words and debounce_ok) or punct:
        _last_partial_text[label] = txt
        _last_partial_time[label] = now
        return True
    return False

def _reset_partial(label: str):
    _last_partial_text[label] = ""
    _last_partial_time[label] = time.monotonic() * 1000.0

# ------------------- Speechmatics helpers -------------------
async def connect_ws(url, headers):
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)

def _text_from_results(d: dict) -> str:
    words = []
    for r in d.get("results", []):
        alts = r.get("alternatives", [])
        if alts:
            w = alts[0].get("content", "")
            if w: words.append(w)
    return " ".join(words).strip()

def _extract_transcript(d: dict) -> str:
    meta_t = (d.get("metadata") or {}).get("transcript") or ""
    res_t  = _text_from_results(d)
    return res_t if len(res_t) > len(meta_t) else meta_t

def _extract_translation(d: dict) -> str:
    meta = d.get("metadata") or {}
    return d.get("translation") or meta.get("translation") or ""

async def start_sm(label: str):
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
    if ENABLE_TRANSLATION:
        transcription_config["translation_config"] = {"target_languages": ["ru"], "enable_partials": True}

    await ws.send(json.dumps({
        "message": "StartRecognition",
        "audio_format": {"type": "raw", "encoding": ENCODING, "sample_rate": SAMPLE_RATE},
        "transcription_config": transcription_config
    }))

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
                        if _should_print_partial(label, txt):
                            await say_line(label, txt, partial=True)
                    else:
                        await say_line(label, txt, partial=False)
                        _reset_partial(label)

                elif m in ("AddPartialTranslation", "AddTranslation"):
                    tr = _extract_translation(data)
                    if not tr:
                        continue
                    lab = f"{label}→RU"
                    if m == "AddPartialTranslation":
                        if _should_print_partial(lab, tr):
                            await say_tr_line(label, tr, partial=True)
                    else:
                        await say_tr_line(label, tr, partial=False)
                        _reset_partial(lab)

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
        await ws.send(chunk)  # бинарные кадры

    async def finish():
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))
        with contextlib.suppress(Exception):
            await ws.close()

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
        while True:
            await asyncio.sleep(1.0)
            nonlocal last_t
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

# ------------------- main -------------------
async def main():
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

    # S2 — что будем захватывать
    if CAPTURE_S2:
        await head_line(f"[INFO] S2_TARGET (режим/имя) = {S2_TARGET or 'AUTO_DEFAULT_SINK'}")
    else:
        await head_line("[INFO] S2 отключён (CAPTURE_S2=0).")

    await head_line("[INFO] В звонке выберите маршрут Bluetooth на телефоне: jupiter-Asp")

    # 1) Поднимаем нужные WS-потоки и ждём RecognitionStarted
    sessions = []  # (name, ws, reader, send, fin, started, err)

    if CAPTURE_S2:
        ws2, r2, send2, fin2, started2, err2 = await start_sm("S2")
        sessions.append(("S2", ws2, r2, send2, fin2, started2, err2))

    if cap_s1:
        ws1, r1, send1, fin1, started1, err1 = await start_sm("S1")
        sessions.append(("S1", ws1, r1, send1, fin1, started1, err1))

    if not sessions:
        await head_line("[ERROR] Нет активных сессий (оба канала отключены). Включите CAPTURE_S1 и/или CAPTURE_S2.")
        return

    reader_tasks = [asyncio.create_task(s[2]()) for s in sessions]
    await asyncio.wait({asyncio.create_task(s[5].wait()) for s in sessions}, return_when=asyncio.ALL_COMPLETED)

    # 2) Квоты
    active = []
    for s in sessions:
        name, ws, r, send, fin, started, err = s
        if err.get("type") == "quota_exceeded":
            await head_line(f"[FALLBACK] {name} отклонён квотой — отключаю.")
            with contextlib.suppress(Exception):
                await fin()
        else:
            active.append(s)

    if not active:
        await head_line("[ERROR] quota_exceeded для всех сессий. Закройте висящие RT-сессии и перезапустите.")
        for t in reader_tasks:
            with contextlib.suppress(Exception): t.cancel()
        return

    # 3) S2: определяем sink
    tasks = []
    have_S2 = any(s[0] == "S2" for s in active)

    actual_s2 = None
    if have_S2:
        mode = S2_TARGET or "AUTO_DEFAULT_SINK"
        if mode in ("AUTO_DEFAULT_SINK", "AUTO_BT_SINK", "AUTO_ALSA_SINK"):
            await info_line(f"[S2] режим {mode} — выбираю соответствующий sink…")
        else:
            await info_line(f"[S2] ожидаемый sink: {mode}")

        actual_s2 = await wait_for_sink(mode, timeout_sec=180)
        if not actual_s2:
            await head_line("[ERROR] Не удалось найти подходящий sink для S2. Подключите наушники/выход и повторите.")
            for s in active:
                with contextlib.suppress(Exception): await s[4]()  # fin
            for t in reader_tasks:
                with contextlib.suppress(Exception): t.cancel()
            return
        await info_line(f"[S2] выбран sink: {actual_s2}")

    # 4) Запускаем захваты
    for s in active:
        name, ws, r, send, fin, started, err = s
        if name == "S2":
            cmd = [
                "pw-record", "--target", actual_s2,
                "--properties", "stream.capture.sink=true",
                "--format", "s16", "--channels", "1",
                "--rate", str(SAMPLE_RATE), "-"
            ]
            tasks.append(asyncio.create_task(run_pw_record(cmd, send, "S2_capture")))
        elif name == "S1":
            cmd = [
                "pw-record", "--target", S1_DEVICE,
                "--format", "s16", "--channels", "1",
                "--rate", str(SAMPLE_RATE), "-"
            ]
            tasks.append(asyncio.create_task(run_pw_record(cmd, send, "S1_capture")))

    # 5) Завершение по Ctrl+C
    stop = asyncio.Event()
    def _sig(*_): stop.set()
    for sgn in (signal.SIGINT, signal.SIGTERM): signal.signal(sgn, _sig)
    await stop.wait()

    # 6) Закрытие
    for s in active:
        with contextlib.suppress(Exception): await s[4]()  # fin
    for t in reader_tasks + tasks:
        with contextlib.suppress(Exception): t.cancel()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass

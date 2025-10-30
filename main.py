#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime ASR + EN↔RU Translation + Simple Two-Speaker Diarisation (Ubuntu, local-only)

Ключевые моменты:
- Whisper грузится ТОЛЬКО из локальной папки (никаких сетевых скачиваний).
- Переводчики HF — только из локального кэша (safetensors), без *.bin.
- Стабильная сегментация речи, фильтрация сомнительных сегментов.
- Подсказки Ollama: health-check + тёплый старт + backoff, без спама ошибками.
"""

import os

# =========================
# КОНФИГУРАЦИЯ
# =========================
# --- Whisper: локальная папка Systran/faster-whisper-<model> ---
ASR_MODEL_DIR       = os.path.expanduser("~/.cache/whisper/large-v3")  # положите сюда 5 файлов модели
RUNTIME_DEVICE      = "cuda"            # "cuda" или "cpu"
COMPUTE_TYPE        = "int8_float16"    # "float16" | "int8" | "int8_float16" | "int8_float32"

# --- Диаризация (простая альтернация S1/S2) ---
DIARIZER_MODE       = "alt"             # "alt" | "none"

# --- Подсказки (Ollama) ---
SUGGESTIONS_ENABLED     = True
OLLAMA_URL              = "http://localhost:11434"
OLLAMA_MODEL            = "mistral:latest"#"llama3.2:1b"   # лёгкая модель для CPU
OLLAMA_TIMEOUT_S        = 7.0             # рабочий таймаут
OLLAMA_WARMUP_TIMEOUT_S = 25.0            # только на тёплый старт
OLLAMA_MAX_TOKENS       = 16
SUGGESTIONS_LOG_ERRORS  = False           # не печатать таймауты в консоль

# --- Аудио ---
DEVICE_INDEX        = 2                   # ваш USB-микрофон
INPUT_SAMPLE_RATE   = 48000
TARGET_ASR_SR       = 16000

# --- VAD / сегментация ---
FRAME_MS            = 20
VAD_AGGR            = 2                   # мягче, чем 3 — меньше ложных «начал»
SILENCE_TAIL_MS     = 500                 # пауза до финализации
MAX_SEGMENT_S       = 9.0
PARTIAL_MIN_MS      = 1100
MIN_ACTIVE_FRAMES   = 4

# --- Калибровка/энергогейт ---
CALIBRATE_SECS      = 0.8
ENERGY_MARGIN_DB    = 6.0                 # +6 dB к медиане фона
ENERGY_DB_MIN       = -55.0
ENERGY_DB_MAX       = -30.0               # не поднимаем порог выше -30 dBFS

# --- Прочее ---
LOG_PATH            = "conversation_log.jsonl"
ENABLE_PARTIAL      = False               # по умолчанию выкл: ASR быстрее/чище
TRANSF_VERBOSITY    = "error"

# HF-кэш (для переводчиков)
CACHE_ROOT = os.path.expanduser("~/.cache/hf")
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
# hf-transfer выключен — он у вас тормозил на больших файлах
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", TRANSF_VERBOSITY)
# Тише логи onnxruntime/ctranslate2
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
os.environ.setdefault("CT2_VERBOSE", "0")

os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

# =========================
# ИМПОРТЫ
# =========================
import json
import queue
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Tuple
from fractions import Fraction
import difflib

import numpy as np
import requests
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from scipy import signal

# =========================
# УТИЛИТЫ
# =========================
def now_utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def pcm16_bytes_to_float32(buf: bytes) -> np.ndarray:
    x = np.frombuffer(buf, dtype=np.int16)
    return x.astype(np.float32) / 32768.0

def rms_dbfs_from_bytes(buf: bytes) -> float:
    if not buf:
        return -120.0
    x = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(x))) / 32768.0 + 1e-12
    return 20.0 * np.log10(rms)

def summarize_for_prompt(history: List[str], last_utt: str, limit: int = 6) -> str:
    ctx = "\n".join(history[-limit:])
    return (
        "You are a concise assistant. Based on the recent conversation, propose ONE short, natural, and helpful English reply.\n"
        "Constraints: <= 16 words. Do not explain. Output only the reply.\n"
        "Conversation:\n"
        f"{ctx}\n"
        f"Last utterance: {last_utt}\n"
        "Reply:"
    )

def _ollama_generate(executor: ThreadPoolExecutor, url: str, model: str, prompt: str,
                     timeout: float, max_tokens: int):
    def _call():
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": max_tokens}
        }
        try:
            r = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            text = (data.get("response") or "").strip()
            return {"ok": bool(text), "text": text or ""}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return executor.submit(_call)

def resample_to(audio_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio_f32.astype(np.float32, copy=False)
    frac = Fraction(dst_sr, src_sr).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return signal.resample_poly(audio_f32, up, down).astype(np.float32, copy=False)

# =========================
# ДИАРИЗАЦИЯ (ALT)
# =========================
class BaseDiarizer:
    def assign(self, audio_f32_16k: np.ndarray) -> Tuple[str, float, dict]:
        return "S1", 0.0, {}
    def hint_partial(self) -> Optional[str]:
        return None

class AltDiarizer(BaseDiarizer):
    def __init__(self):
        self._next = "S1"
        self.last_speaker = None
    def assign(self, audio_f32_16k: np.ndarray) -> Tuple[str, float, dict]:
        s = self._next
        self._next = "S2" if self._next == "S1" else "S1"
        self.last_speaker = s
        return s, 0.5, {"mode": "alt"}
    def hint_partial(self) -> Optional[str]:
        return self.last_speaker

# =========================
# Переводчики: локально, только safetensors
# =========================
def _build_translation_pipeline_local(repo_id: str):
    from huggingface_hub import snapshot_download
    allow = ["*.safetensors","*.json","*.txt","*.model",
             "tokenizer.json","tokenizer_config.json",
             "config.json","vocab.json","merges.txt","*.spm",
             "special_tokens_map.json","source.spm","target.spm"]
    ignore = ["*.bin","pytorch_model.bin","rust_model.ot"]

    local_dir = snapshot_download(repo_id=repo_id,
                                  local_files_only=True,
                                  allow_patterns=allow,
                                  ignore_patterns=ignore)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True)
    return pipeline("translation", model=mdl, tokenizer=tok, device=-1)

# =========================
# ОСНОВНОЙ КЛАСС
# =========================
class RealtimeASRTranslate:
    def __init__(
        self,
        asr_model_dir: str = ASR_MODEL_DIR,
        compute_type: str = COMPUTE_TYPE,
        device: str = RUNTIME_DEVICE,
        diarizer_mode: str = DIARIZER_MODE,
        log_path: str = LOG_PATH,
        enable_partial: bool = ENABLE_PARTIAL,
        device_index: Optional[int] = DEVICE_INDEX,
        frame_ms: int = FRAME_MS,
        vad_aggr: int = VAD_AGGR,
        silence_tail_ms: int = SILENCE_TAIL_MS,
        max_segment_s: float = MAX_SEGMENT_S,
        partial_min_ms: int = PARTIAL_MIN_MS,
        input_samplerate: Optional[int] = INPUT_SAMPLE_RATE,
        target_asr_sr: int = TARGET_ASR_SR,
    ):
        # Проверим, что локальная папка Whisper содержит нужные файлы
        need = ["model.bin","tokenizer.json","vocabulary.json","config.json","preprocessor_config.json"]
        missing = [f for f in need if not os.path.isfile(os.path.join(asr_model_dir, f))]
        if missing:
            raise RuntimeError(
                f"В каталоге {asr_model_dir} нет обязательных файлов: {', '.join(missing)}.\n"
                f"Скопируйте их из репо Systran/faster-whisper-<model> (локально)."
            )

        self.device_index = device_index
        self.in_sr = int(input_samplerate)
        self.target_sr = int(target_asr_sr)

        # ASR: только локально
        self._whisper = WhisperModel(
            asr_model_dir,
            device=device,
            compute_type=compute_type
        )

        # Переводчики (ленивая загрузка)
        self._mt_enru = None
        self._mt_ruen = None

        # Ollama
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_MODEL
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._ollama_ok = self._init_ollama() if SUGGESTIONS_ENABLED else False
        self._sugg_backoff_until = 0.0
        self._sugg_backoff = 0.0  # сек

        # VAD/тайминги
        self.vad = webrtcvad.Vad(vad_aggr)
        self.frame_ms = int(frame_ms)
        self.frame_bytes = int(self.in_sr * (self.frame_ms / 1000.0)) * 2 * 1
        self.silence_tail_frames = int(silence_tail_ms / self.frame_ms)
        self.partial_min_frames = int(partial_min_ms / self.frame_ms)
        self.max_segment_frames = int((max_segment_s * 1000.0) / self.frame_ms)
        self.min_active_frames = int(MIN_ACTIVE_FRAMES)

        # Очереди / состояние
        self.audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=256)
        self.enable_partial = enable_partial
        self.history_lines: "deque[str]" = deque(maxlen=64)
        self._last_partial_text: str = ""
        self._last_final_text: str = ""
        self.diarizer = AltDiarizer() if (diarizer_mode or "alt").lower().strip() != "none" else BaseDiarizer()
        self.energy_threshold_db = -38.0
        self._last_overflow_log_ts = 0.0
        self.running = False

        print(f"[INFO] Whisper local path: {asr_model_dir}")
        print(f"[INFO] Input device index: {self.device_index}, in_sr={self.in_sr} Hz; ASR target_sr={self.target_sr} Hz", flush=True)

    # ---- Ollama init: health-check + тёплый старт
    def _init_ollama(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3.0)
            r.raise_for_status()
            # тёплый старт (один короткий вызов, увеличенный таймаут)
            warm = _ollama_generate(self._executor, self.ollama_url, self.ollama_model,
                                    "Say: OK", timeout=OLLAMA_WARMUP_TIMEOUT_S, max_tokens=3)
            res = warm.result()
            ok = res.get("ok", False)
            if ok:
                print(f"[SUGGEST] Ollama OK, модель готова: {self.ollama_model}", flush=True)
            else:
                if SUGGESTIONS_LOG_ERRORS:
                    print(f"[SUGGEST] Warm-up failed: {res.get('error','unknown')}", flush=True)
            return True
        except Exception as e:
            if SUGGESTIONS_LOG_ERRORS:
                print(f"[SUGGEST] Ollama недоступен ({e}). Подсказки будут выключены.", flush=True)
            return False

    # Локальные переводчики (CPU)
    def _translator_enru(self):
        if self._mt_enru is None:
            self._mt_enru = _build_translation_pipeline_local("Helsinki-NLP/opus-mt-en-ru")
        return self._mt_enru

    def _translator_ruen(self):
        if self._mt_ruen is None:
            self._mt_ruen = _build_translation_pipeline_local("Helsinki-NLP/opus-mt-ru-en")
        return self._mt_ruen

    # PortAudio callback
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            now = time.time()
            if "input_overflow" in str(status) and (now - self._last_overflow_log_ts) > 2.0:
                print("[VAD] PortAudio input_overflow", flush=True)
                self._last_overflow_log_ts = now
        try:
            self.audio_q.put_nowait(bytes(indata))
        except queue.Full:
            _ = self.audio_q.get_nowait()
            self.audio_q.put_nowait(bytes(indata))

    # Калибровка шумового порога
    def _calibrate_noise(self, seconds: float = CALIBRATE_SECS):
        frames_to_collect = max(10, int(round((seconds * 1000.0) / self.frame_ms)))
        print(f"[INFO] Calibrating noise for ~{seconds:.1f}s ({frames_to_collect} frames)...", flush=True)

        vals = []
        deadline = time.time() + max(2.5, seconds * 3.0)
        while len(vals) < frames_to_collect and time.time() < deadline:
            try:
                frame = self.audio_q.get(timeout=0.4)
            except queue.Empty:
                continue
            if len(frame) != self.frame_bytes:
                continue
            vals.append(rms_dbfs_from_bytes(frame))

        if vals:
            import numpy as _np
            med = float(_np.median(vals))
            thr = med + ENERGY_MARGIN_DB
            thr = max(min(thr, ENERGY_DB_MAX), ENERGY_DB_MIN)
            self.energy_threshold_db = thr
            print(f"[INFO] Noise calibration: median={med:.1f} dBFS -> threshold={self.energy_threshold_db:.1f} dBFS", flush=True)
        else:
            print(f"[WARN] Noise calibration: no frames, using default threshold {self.energy_threshold_db:.1f} dBFS", flush=True)

    # Основной цикл
    def run(self):
        self.running = True
        blocksize = int(self.in_sr * (self.frame_ms / 1000.0))

        print("[INFO] Opening input stream...", flush=True)
        with sd.RawInputStream(
            samplerate=self.in_sr,
            blocksize=blocksize,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
            device=self.device_index
        ):
            self._calibrate_noise(CALIBRATE_SECS)
            print(f"[{now_utc_iso()}] Listening (Ctrl+C to stop)...", flush=True)

            in_speech = False
            silence_count = 0
            active_streak = 0
            frames_in_segment = 0
            voiced_frames: List[bytes] = []
            last_partial_t = time.time()

            while self.running:
                try:
                    frame = self.audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if len(frame) != self.frame_bytes:
                    continue

                db = rms_dbfs_from_bytes(frame)
                energetic = (db >= self.energy_threshold_db)
                is_speech = energetic and self.vad.is_speech(frame, self.in_sr)

                if is_speech:
                    active_streak += 1
                    if not in_speech:
                        if active_streak >= self.min_active_frames:
                            in_speech = True
                            silence_count = 0
                            frames_in_segment = 0
                            voiced_frames = []
                    if in_speech:
                        voiced_frames.append(frame)
                        frames_in_segment += 1

                        if self.enable_partial and frames_in_segment >= (PARTIAL_MIN_MS // self.frame_ms):
                            now_ = time.time()
                            if now_ - last_partial_t > 0.9:
                                last_partial_t = now_
                                self._process_segment(voiced_frames, finalize=False)
                else:
                    active_streak = 0
                    if in_speech:
                        silence_count += 1
                        if silence_count >= self.silence_tail_frames or frames_in_segment >= self.max_segment_frames:
                            self._process_segment(voiced_frames, finalize=True)
                            in_speech = False
                            voiced_frames = []
                            frames_in_segment = 0
                            silence_count = 0

    # Декодирование + фильтр сомнительных сегментов
    def _decode_and_filter(self, audio_16k: np.ndarray):
        segments, info = self._whisper.transcribe(
            audio_16k,
            task="transcribe",
            language=None,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_speech_duration_ms": 400,
                            "max_speech_duration_s": 16,
                            "min_silence_duration_ms": 500},
            no_speech_threshold=0.85,
            condition_on_previous_text=False,
        )

        chosen = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if not txt:
                continue
            dur = (seg.end - seg.start) if (seg.end is not None and seg.start is not None) else 0.0
            avg_lp = getattr(seg, "avg_logprob", None)
            nsp = getattr(seg, "no_speech_prob", None)

            # Отсечём всё с низкой уверенностью / слишком короткое
            if avg_lp is not None and avg_lp < -1.2:
                continue
            if nsp is not None and nsp > 0.6 and dur < 1.0:
                continue
            if len(txt) <= 2 and (avg_lp is None or avg_lp < -0.6):
                continue

            chosen.append(txt)

        text = " ".join(chosen).strip()
        lang = str(getattr(info, "language", "") or "")
        return text, lang

    # Обработка сегмента
    def _process_segment(self, frames_list: List[bytes], finalize: bool):
        if not frames_list:
            return
        pcm = b"".join(frames_list)
        audio_in = pcm16_bytes_to_float32(pcm)
        audio_16k = resample_to(audio_in, self.in_sr, self.target_sr)

        text, lang = self._decode_and_filter(audio_16k)
        if not text:
            return

        # Дедупликация
        if not finalize:
            ratio = difflib.SequenceMatcher(None, self._last_partial_text, text).ratio()
            if text == self._last_partial_text or ratio > 0.94:
                return
            self._last_partial_text = text
        else:
            if text == self._last_final_text:
                return
            self._last_final_text = text
            self._last_partial_text = ""

        src_lang = "en" if lang.startswith("en") else ("ru" if lang.startswith("ru") else "unk")

        # Перевод только для финала
        if finalize:
            if src_lang == "en":
                translation = self._translator_enru()(text)[0]["translation_text"]; tgt_lang = "ru"
            elif src_lang == "ru":
                translation = self._translator_ruen()(text)[0]["translation_text"]; tgt_lang = "en"
            else:
                if all((ord(c) < 128) for c in text):
                    translation = self._translator_enru()(text)[0]["translation_text"]; src_lang, tgt_lang = "en", "ru"
                else:
                    translation = self._translator_ruen()(text)[0]["translation_text"]; src_lang, tgt_lang = "ru", "en"
        else:
            translation, tgt_lang = "", ""

        # Диаризация
        if finalize:
            speaker, spk_conf, spk_extra = self.diarizer.assign(audio_16k)
        else:
            speaker, spk_conf, spk_extra = (self.diarizer.hint_partial() or "UNK"), 0.0, {"mode": "hint"}

        rec = {
            "ts": now_utc_iso(), "final": bool(finalize),
            "speaker": speaker, "speaker_conf": float(spk_conf),
            "speaker_info": spk_extra,
            "src_lang": src_lang, "tgt_lang": tgt_lang,
            "text": text, "translation": translation
        }
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

        label = "FINAL" if finalize else "PARTIAL"
        if finalize:
            print(f"[{rec['ts']}] ({label}) {speaker} [{src_lang}->{tgt_lang}]: {text}")
            print(f"   └─ {translation}", flush=True)
        else:
            print(f"[{rec['ts']}] ({label}) {speaker}: {text}", flush=True)

        # Подсказка: с backoff, без спама
        if finalize and SUGGESTIONS_ENABLED and self._ollama_ok:
            now_ = time.time()
            if now_ >= self._sugg_backoff_until:
                self.history_lines.append(f"{speaker}: {text}")
                prompt = summarize_for_prompt(list(self.history_lines), text)
                fut = _ollama_generate(self._executor, self.ollama_url, self.ollama_model,
                                       prompt, timeout=OLLAMA_TIMEOUT_S, max_tokens=OLLAMA_MAX_TOKENS)

                def _on_done(f):
                    res = f.result()
                    if res.get("ok"):
                        self._sugg_backoff = 0.0
                        self._sugg_backoff_until = 0.0
                        suggestion = res["text"]
                        try:
                            with open(LOG_PATH, "a", encoding="utf-8") as ff:
                                ff.write(json.dumps({"ts": now_utc_iso(), "type": "suggestion",
                                                     "model": self.ollama_model, "to_say_en": suggestion},
                                                    ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        print(f"   ⇒ Suggested reply (EN): {suggestion}", flush=True)
                    else:
                        if SUGGESTIONS_LOG_ERRORS:
                            err = res.get("error", "unknown_error")
                            print(f"   ⇒ Suggested reply (EN): [error: {err}]", flush=True)
                        # backoff: 6s → 18s → 54s …
                        self._sugg_backoff = 6.0 if self._sugg_backoff == 0.0 else min(self._sugg_backoff * 3.0, 180.0)
                        self._sugg_backoff_until = time.time() + self._sugg_backoff

                fut.add_done_callback(_on_done)

# =========================
# ТОЧКА ВХОДА
# =========================
def main():
    app = RealtimeASRTranslate()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()

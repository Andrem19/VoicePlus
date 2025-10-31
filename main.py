#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime ASR (local faster-whisper) + Robust 2-speaker diarisation (online ECAPA + smoothing) + optional async EN↔RU translation.

Goals now:
- PRIORITY: capture every utterance quickly and assign the right speaker (S1/S2) with minimal lag.
- Translation runs in a separate thread and NEVER blocks ASR output. You can turn it off.
- No LLM suggestions in this build.

What changed vs previous version:
- Stronger external VAD and stricter no-speech filtering ⇒ fewer “phantom” lines in silence.
- Online ECAPA diariser on CPU with:
    • two prototype centroids (S1/S2) that adapt slowly;
    • a small pending queue: we re-classify the last few segments once both prototypes exist, then flush,
      which stabilises “who is who” without noticeable delay (~0.6–1.0 s).
    • hysteresis + cooldown to avoid S1↔S2 flip-flops.
- Anti-fragmentation merge only after diariser stabilizes; short back-to-back pieces from the same speaker are merged.
- Removed initial_prompt to reduce hallucinations (“contact us”, etc.).
- Higher no_speech_threshold and min segment duration; plus lexical/length gate to drop junk.

Local files required (no network):
  Whisper dir (~/.cache/whisper/large-v3):
    model.bin, tokenizer.json, vocabulary.json, config.json, preprocessor_config.json
  (Optional) Translators in local HF cache:
    Helsinki-NLP/opus-mt-en-ru, Helsinki-NLP/opus-mt-ru-en (safetensors only)
  (Optional) Speaker model in local HF cache:
    speechbrain/spkrec-ecapa-voxceleb

Console commands:
  :me s1 | :me s2     set who is YOU (for your own reference in logs)
  :swap               swap who is YOU
  :translate on|off   enable/disable async translation
"""

import os

# =========================
# CONFIG
# =========================
ASR_MODEL_DIR              = os.path.expanduser("~/.cache/whisper/large-v3")  # 5 files from Systran/faster-whisper-<model>
DEVICE                     = "cuda"       # "cuda" | "cpu"
COMPUTE_TYPE               = "float16"    # "int8_float16" if very low VRAM

# Whisper decode tuning (fast + stable)
WHISPER_BEAM_SIZE          = 2
WHISPER_TEMP               = 0.0
WHISPER_WITHOUT_TS         = True
WHISPER_CONDITION_PREV     = False
WHISPER_INTERNAL_VAD       = False        # we use external VAD
WHISPER_NO_SPEECH_TH       = 0.80         # stricter: drop more silence
WHISPER_COMP_RATIO_TH      = 2.4          # default guard
WHISPER_LOGPROB_TH         = -1.0         # default guard

# VAD / segmentation (balance completeness vs speed)
FRAME_MS                   = 20
VAD_AGGR                   = 3            # stricter; fewer false starts
SILENCE_TAIL_MS            = 800          # longer tail → fewer clipped endings
MAX_SEGMENT_S              = 14.0
MIN_SEGMENT_MS             = 1400
MIN_ACTIVE_FRAMES          = 5

# Noise gate calibration
CALIBRATE_SECS             = 0.8
ENERGY_MARGIN_DB           = 6.0
ENERGY_DB_MIN              = -50.0
ENERGY_DB_MAX              = -30.0

# Merge behaviour (same-speaker, back-to-back)
MERGE_GAP_S_MAX            = 1.4
MERGE_WINDOW_S_MAX         = 9.0
MERGE_WORDS_MAX            = 7

# Pending queue (diariser stabilisation)
PENDING_MAX_ITEMS          = 6
PENDING_MAX_AGE_S          = 1.0

# Diarisation (ECAPA online, CPU)
USE_ECAPA_IF_AVAILABLE     = True
REASSIGN_COOLDOWN_S        = 0.80
SIM_THRESHOLD_NEW          = 0.55
SIM_DELTA_STABLE           = 0.05
PROTOTYPE_MOMENTUM         = 0.10

# Translation (async, non-blocking)
TRANSLATE_ENABLED_DEFAULT  = True
TRANSLATE_TIMEOUT_S        = 3.0

# Audio device
DEVICE_INDEX               = 2
INPUT_SR                   = 48000
ASR_SR                     = 16000

LOG_PATH                   = "conversation_log.jsonl"

# HF caches (strictly local)
CACHE_ROOT = os.path.expanduser("~/.cache/hf")
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
os.environ.setdefault("CT2_VERBOSE", "0")
os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

# =========================
# IMPORTS
# =========================
import re
import sys
import time
import json
import math
import queue
import threading
from typing import Optional, Tuple, List, Deque, Dict, Any
from collections import deque
from fractions import Fraction

import numpy as np
import sounddevice as sd
import webrtcvad
from scipy import signal
from faster_whisper import WhisperModel

# Optional modules
_TRANSLATORS_OK = False
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

_HF_AVAILABLE = False
try:
    from huggingface_hub import snapshot_download
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

_SPEECHBRAIN_OK = False
try:
    import torch
    # v1.0 moved to speechbrain.inference
    try:
        from speechbrain.inference import EncoderClassifier
    except Exception:
        from speechbrain.pretrained import EncoderClassifier  # fallback; may print a deprecation warning
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


# =========================
# UTILS
# =========================
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

def pcm16_to_f32(buf: bytes) -> np.ndarray:
    x = np.frombuffer(buf, dtype=np.int16)
    return (x.astype(np.float32) / 32768.0)

def rms_dbfs(buf: bytes) -> float:
    if not buf: return -120.0
    x = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    if x.size == 0: return -120.0
    rms = np.sqrt(np.mean(np.square(x))) / 32768.0 + 1e-12
    return 20.0 * np.log10(rms)

def resample(audio_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr: return audio_f32.astype(np.float32, copy=False)
    frac = Fraction(dst_sr, src_sr).limit_denominator(1000)
    return signal.resample_poly(audio_f32, frac.numerator, frac.denominator).astype(np.float32, copy=False)

_SHORT_ALLOW = {"yes", "yeah", "yep", "no", "nope", "thanks", "thank you", "please", "okay", "ok", "alright"}
def looks_like_junk(text: str) -> bool:
    if not text: return True
    t = text.strip().lower()
    # drop 1–2 token generic noise unless allowlisted
    if len(t.split()) <= 2 and t not in _SHORT_ALLOW:
        # if it's clearly address/postcode/digits, keep
        if re.search(r"\b\d+[a-z]?\b", t) or re.search(r"\b[a-z]{1,2}\d[a-z0-9]?\s*\d[a-z]{2}\b", t, re.I):
            return False
        return True
    return False

def normalize_postcode(text: str) -> str:
    # Normalize UK postcode like FY4 3HX
    m = re.search(r"\b([A-Z]{1,2}\d[A-Z0-9]?\s*\d[A-Z]{2})\b", text.upper())
    return (text if not m else text.replace(m.group(0), re.sub(r"\s+", " ", m.group(1))))


# =========================
# DIARISATION
# =========================
class DiariserBase:
    def assign(self, audio_16k: np.ndarray, seg_ts: float) -> str:
        return "S1"

class TurnTakingDiariser(DiariserBase):
    def __init__(self):
        self.next_spk = "S1"
    def assign(self, audio_16k: np.ndarray, seg_ts: float) -> str:
        s = self.next_spk
        self.next_spk = "S2" if self.next_spk == "S1" else "S1"
        return s

class ECAPADiariser(DiariserBase):
    """Two-centroid online diariser with hysteresis + cooldown."""
    def __init__(self, classifier: "EncoderClassifier"):
        self.classifier = classifier
        self.p1: Optional[np.ndarray] = None
        self.p2: Optional[np.ndarray] = None
        self.last_spk: Optional[str] = None
        self.last_ts: float = 0.0

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _embed(self, audio_16k: np.ndarray) -> np.ndarray:
        # limit to ~3s middle to reduce noise; always CPU to avoid GPU contention
        sr = 16000
        T = len(audio_16k)
        target = 3 * sr
        if T > target:
            start = (T - target) // 2
            audio_16k = audio_16k[start:start+target]
        with torch.no_grad():
            wav = torch.from_numpy(audio_16k).float().unsqueeze(0)
            emb = self.classifier.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb

    def assign(self, audio_16k: np.ndarray, seg_ts: float) -> Tuple[str, Dict[str, float]]:
        emb = self._embed(audio_16k)
        # Cooldown: keep previous to avoid rapid flips
        if self.last_spk and (seg_ts - self.last_ts) < REASSIGN_COOLDOWN_S:
            spk = self.last_spk
            self._adapt(spk, emb)
            self.last_ts = seg_ts
            return spk, {"cooldown": 1.0}

        if self.p1 is None:
            self.p1 = emb; self.last_spk = "S1"; self.last_ts = seg_ts
            return "S1", {"bootstrap": 1.0}

        if self.p2 is None:
            sim1 = self._cos(emb, self.p1)
            if sim1 >= SIM_THRESHOLD_NEW:
                self._adapt("S1", emb)
                self.last_spk = "S1"; self.last_ts = seg_ts
                return "S1", {"sim1": sim1}
            else:
                self.p2 = emb; self.last_spk = "S2"; self.last_ts = seg_ts
                return "S2", {"new_spk": 1.0}

        sim1 = self._cos(emb, self.p1)
        sim2 = self._cos(emb, self.p2)
        if abs(sim1 - sim2) <= SIM_DELTA_STABLE and self.last_spk is not None:
            spk = self.last_spk
        else:
            spk = "S1" if sim1 >= sim2 else "S2"

        self._adapt(spk, emb)
        self.last_spk = spk; self.last_ts = seg_ts
        return spk, {"sim1": sim1, "sim2": sim2}

    def _adapt(self, spk: str, emb: np.ndarray):
        if spk == "S1":
            self.p1 = (1 - PROTOTYPE_MOMENTUM) * self.p1 + PROTOTYPE_MOMENTUM * emb
            self.p1 /= (np.linalg.norm(self.p1) + 1e-9)
        else:
            self.p2 = (1 - PROTOTYPE_MOMENTUM) * self.p2 + PROTOTYPE_MOMENTUM * emb
            self.p2 /= (np.linalg.norm(self.p2) + 1e-9)


# =========================
# TRANSLATION (async & optional)
# =========================
class AsyncTranslator:
    def __init__(self):
        self.enabled = TRANSLATE_ENABLED_DEFAULT and _TRANSFORMERS_AVAILABLE
        self._enru = None
        self._ruen = None
        self.pool = None
        if self.enabled:
            try:
                self._enru = pipeline(
                    "translation",
                    model=AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru", local_files_only=True),
                    tokenizer=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru", local_files_only=True),
                    device=-1,
                )
                self._ruen = pipeline(
                    "translation",
                    model=AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en", local_files_only=True),
                    tokenizer=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en", local_files_only=True),
                    device=-1,
                )
                from concurrent.futures import ThreadPoolExecutor
                self.pool = ThreadPoolExecutor(max_workers=2)
                global _TRANSLATORS_OK
                _TRANSLATORS_OK = True
            except Exception:
                self.enabled = False

    def translate_now(self, speaker: str, text: str, src_hint: str):
        if not (self.enabled and _TRANSLATORS_OK and self.pool and text):
            return None
        def _task():
            try:
                if src_hint.startswith("en") or all(ord(c) < 128 for c in text):
                    t = self._enru(text, truncation=True, max_length=256)[0]["translation_text"]
                    return ("ru", t)
                else:
                    t = self._ruen(text, truncation=True, max_length=256)[0]["translation_text"]
                    return ("en", t)
            except Exception:
                return None
        from concurrent.futures import wait, FIRST_COMPLETED
        fut = self.pool.submit(_task)
        done, _ = wait([fut], timeout=TRANSLATE_TIMEOUT_S, return_when=FIRST_COMPLETED)
        if fut in done:
            return fut.result()
        try:
            fut.cancel()
        except Exception:
            pass
        return None


# =========================
# APP
# =========================
class RealtimeASRApp:
    def __init__(self):
        # Check whisper files
        need = ["model.bin","tokenizer.json","vocabulary.json","config.json","preprocessor_config.json"]
        miss = [f for f in need if not os.path.isfile(os.path.join(ASR_MODEL_DIR, f))]
        if miss:
            raise RuntimeError(f"Missing in {ASR_MODEL_DIR}: {', '.join(miss)}")

        t0 = time.time()
        self.whisper = WhisperModel(ASR_MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"[INIT] Whisper loaded in {time.time()-t0:.2f}s (device={DEVICE}, compute={COMPUTE_TYPE})", flush=True)

        # Diariser: prefer ECAPA if local
        self.diariser: DiariserBase
        if USE_ECAPA_IF_AVAILABLE and _HF_AVAILABLE and _TORCH_OK:
            try:
                snapshot_download("speechbrain/spkrec-ecapa-voxceleb", local_files_only=True)
                self.diariser = ECAPADiariser(EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                    savedir=os.path.join(os.environ["HUGGINGFACE_HUB_CACHE"], "speechbrain_spkrec_ecapa_voxceleb_local")
                ))
                global _SPEECHBRAIN_OK
                _SPEECHBRAIN_OK = True
                print("[INIT] ECAPA diariser ready (local).", flush=True)
            except Exception as e:
                print(f"[WARN] ECAPA diariser not available locally ({e}). Falling back to turn-taking.", flush=True)
                self.diariser = TurnTakingDiariser()
        else:
            self.diariser = TurnTakingDiariser()

        # Translator
        self.translator = AsyncTranslator()
        if self.translator.enabled:
            print("[INIT] Translators ready (local).", flush=True)
        else:
            print("[INIT] Translators disabled or not local; ASR will not wait for them.", flush=True)

        # Audio & VAD
        self.in_sr = int(INPUT_SR)
        self.asr_sr = int(ASR_SR)
        self.frame_ms = int(FRAME_MS)
        self.frame_bytes = int(self.in_sr * (self.frame_ms / 1000.0)) * 2
        self.vad = webrtcvad.Vad(int(VAD_AGGR))
        self.silence_tail_frames = int(SILENCE_TAIL_MS / self.frame_ms)
        self.min_segment_frames = int(MIN_SEGMENT_MS / self.frame_ms)
        self.max_segment_frames = int((MAX_SEGMENT_S * 1000.0) / self.frame_ms)
        self.min_active_frames = int(MIN_ACTIVE_FRAMES)
        self.audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=256)

        # State
        self.energy_threshold_db = -40.0
        self.last_overflow_log_ts = 0.0
        self.running = False

        # Pending list (for diariser stabilisation BEFORE printing)
        self.pending: Deque[Dict[str, Any]] = deque()  # each: {ts, audio16, text, spk?, printed}
        # Merge buffer (after we decide the speaker)
        self.buf_spk: Optional[str] = None
        self.buf_text: str = ""
        self.buf_start_ts: float = 0.0
        self.last_flush_ts: float = 0.0

        # Role mapping (for your reference)
        self.me_spk: Optional[str] = None

        self.log_path = LOG_PATH

        # Console thread
        self.ctrl = threading.Thread(target=self.console_loop, daemon=True)
        self.ctrl.start()

    # ---------- console ----------
    def console_loop(self):
        print("[ROLE] Type ':me s1' or ':me s2'. ':swap' to swap. ':translate on|off' to toggle translation.", flush=True)
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.05); continue
                line = line.strip()
                if not line: continue
                l = line.lower()
                if l.startswith(":me "):
                    arg = l[4:].strip()
                    if arg in ("s1","1"): self.me_spk = "S1"; print("[CTRL] ME = S1", flush=True)
                    elif arg in ("s2","2"): self.me_spk = "S2"; print("[CTRL] ME = S2", flush=True)
                    else: print("[CTRL] Usage: :me s1 | :me s2", flush=True)
                elif l == ":swap":
                    if self.me_spk in ("S1","S2"):
                        self.me_spk = "S2" if self.me_spk == "S1" else "S1"
                        print(f"[CTRL] Swapped. ME = {self.me_spk}", flush=True)
                    else:
                        print("[CTRL] ME not set.", flush=True)
                elif l.startswith(":translate "):
                    arg = l.split(None, 1)[1].strip()
                    if arg in ("on","true","1"):
                        self.translator.enabled = True
                        print("[CTRL] Translation: ON", flush=True)
                    elif arg in ("off","false","0"):
                        self.translator.enabled = False
                        print("[CTRL] Translation: OFF", flush=True)
                    else:
                        print("[CTRL] Use :translate on|off", flush=True)
                else:
                    print("[CTRL] Unknown command.", flush=True)
            except Exception:
                time.sleep(0.1)

    # ---------- audio ----------
    def audio_cb(self, indata, frames, time_info, status):
        if status:
            now = time.time()
            if "input_overflow" in str(status) and (now - self.last_overflow_log_ts) > 2.0:
                print("[AUDIO] input_overflow", flush=True)
                self.last_overflow_log_ts = now
        try:
            self.audio_q.put_nowait(bytes(indata))
        except queue.Full:
            _ = self.audio_q.get_nowait()
            self.audio_q.put_nowait(bytes(indata))

    # ---------- calibration ----------
    def calibrate_noise(self):
        frames_need = max(12, int(round((CALIBRATE_SECS * 1000.0) / self.frame_ms)))
        print(f"[INFO] Calibrating noise ~{CALIBRATE_SECS:.1f}s ({frames_need} frames)...", flush=True)
        vals = []
        deadline = time.time() + max(2.5, CALIBRATE_SECS * 3)
        while len(vals) < frames_need and time.time() < deadline:
            try:
                fr = self.audio_q.get(timeout=0.4)
            except queue.Empty:
                continue
            if len(fr) != self.frame_bytes: continue
            vals.append(rms_dbfs(fr))
        if vals:
            med = float(np.median(vals))
            thr = max(min(med + ENERGY_MARGIN_DB, ENERGY_DB_MAX), ENERGY_DB_MIN)
            self.energy_threshold_db = thr
            print(f"[INFO] Noise calibration: median={med:.1f} dBFS -> threshold={self.energy_threshold_db:.1f} dBFS", flush=True)
        else:
            print(f"[WARN] Calibration failed; using default {self.energy_threshold_db:.1f} dBFS", flush=True)

    # ---------- ASR ----------
    def asr_decode(self, audio_16k: np.ndarray) -> str:
        segs, info = self.whisper.transcribe(
            audio_16k,
            task="transcribe",
            language="en",
            beam_size=max(1, int(WHISPER_BEAM_SIZE)),
            temperature=WHISPER_TEMP,
            without_timestamps=bool(WHISPER_WITHOUT_TS),
            vad_filter=bool(WHISPER_INTERNAL_VAD),
            condition_on_previous_text=bool(WHISPER_CONDITION_PREV),
            no_speech_threshold=WHISPER_NO_SPEECH_TH,
            compression_ratio_threshold=WHISPER_COMP_RATIO_TH,
            log_prob_threshold=WHISPER_LOGPROB_TH,
            patience=1.0,
            initial_prompt=None,
        )
        text = "".join((s.text or "") for s in segs).strip()
        text = normalize_postcode(text)
        return text

    # ---------- pending / flush ----------
    def _print_and_log(self, spk: str, text: str):
        ts = now_iso()
        print(f"[{ts}] (FINAL) {spk} [en]: {text}", flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": ts, "speaker": spk, "lang": "en", "text": text}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        if self.translator.enabled:
            res = self.translator.translate_now(spk, text, "en")
            if res:
                tgt, tr = res
                print(f"   └─ {tr}", flush=True)
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"ts": now_iso(), "type": "translation", "speaker": spk,
                                            "src_lang": "en", "tgt_lang": tgt, "text": text,
                                            "translation": tr}, ensure_ascii=False) + "\n")
                except Exception:
                    pass

    def _flush_merge_buf(self):
        if not self.buf_text:
            return
        text = self.buf_text.strip()
        spk = self.buf_spk or "S1"
        self.buf_text = ""
        self.buf_spk = None
        self.buf_start_ts = 0.0
        self.last_flush_ts = time.time()
        if text:
            self._print_and_log(spk, text)

    def _merge_or_emit(self, spk: str, text: str):
        now = time.time()
        if self.buf_text and self.buf_spk and self.buf_spk != spk:
            self._flush_merge_buf()

        if not self.buf_text:
            self.buf_spk = spk
            self.buf_text = text
            self.buf_start_ts = now
            if not self._text_incomplete(text):
                self._flush_merge_buf()
            return

        if self.buf_spk == spk:
            # same speaker: try to merge if within window
            if (now - self.buf_start_ts) <= MERGE_WINDOW_S_MAX and (now - self.last_flush_ts) <= MERGE_GAP_S_MAX:
                merged = (self.buf_text + " " + text).strip()
                self.buf_text = merged
                if not self._text_incomplete(merged) or (now - self.buf_start_ts) > MERGE_WINDOW_S_MAX:
                    self._flush_merge_buf()
                return
            # otherwise flush and start new
            self._flush_merge_buf()
            self.buf_spk = spk
            self.buf_text = text
            self.buf_start_ts = now
            if not self._text_incomplete(text):
                self._flush_merge_buf()
            return

        # fallback
        self._flush_merge_buf()
        self.buf_spk = spk
        self.buf_text = text
        self.buf_start_ts = now
        if not self._text_incomplete(text):
            self._flush_merge_buf()

    @staticmethod
    def _text_incomplete(text: str) -> bool:
        if not text: return True
        if len(text.split()) <= MERGE_WORDS_MAX and not re.search(r"[.!?…]\s*$", text):
            return True
        return False

    # ---------- diariser stabilisation ----------
    def _assign_speaker(self, audio_16k: np.ndarray, ts: float) -> str:
        if _SPEECHBRAIN_OK and isinstance(self.diariser, ECAPADiariser):
            spk, _ = self.diariser.assign(audio_16k, ts)
            return spk
        else:
            return self.diariser.assign(audio_16k, ts)

    def _reclassify_pending_if_ready(self):
        """When both prototypes exist, reclassify the last few pending items to stabilise S1/S2."""
        if not (_SPEECHBRAIN_OK and isinstance(self.diariser, ECAPADiariser)):
            return
        d: ECAPADiariser = self.diariser
        if d.p1 is None or d.p2 is None:
            return
        # Reclassify recent items (except the very last to keep latency tiny)
        n = len(self.pending)
        if n <= 1:
            return
        # Reclassify all but the newest
        for i in range(max(0, n - PENDING_MAX_ITEMS), n - 1):
            item = self.pending[i]
            if item.get("printed"):
                continue
            emb_spk, _ = d.assign(item["audio16"], item["ts"])
            item["spk"] = emb_spk

    def _flush_old_pending(self):
        """Flush items older than PENDING_MAX_AGE_S or when queue grows too large."""
        now = time.time()
        while self.pending:
            item = self.pending[0]
            if item.get("printed"):
                self.pending.popleft()
                continue
            age = now - item["ts"]
            if age < PENDING_MAX_AGE_S and len(self.pending) <= PENDING_MAX_ITEMS:
                break
            # Emit through merge buffer
            spk = item.get("spk") or "S1"
            txt = item["text"]
            self._merge_or_emit(spk, txt)
            item["printed"] = True
            self.pending.popleft()

    # ---------- main loop ----------
    def run(self):
        self.running = True
        blocksize = int(self.in_sr * (self.frame_ms / 1000.0))

        print(f"[INFO] Whisper local path: {ASR_MODEL_DIR}", flush=True)
        print(f"[INFO] Input device index: {DEVICE_INDEX}, in_sr={self.in_sr} Hz; ASR target_sr={self.asr_sr} Hz", flush=True)

        with sd.RawInputStream(
            samplerate=self.in_sr,
            blocksize=blocksize,
            dtype="int16",
            channels=1,
            callback=self.audio_cb,
            device=DEVICE_INDEX
        ):
            self.calibrate_noise()
            print(f"[{now_iso()}] Listening (Ctrl+C to stop)...", flush=True)

            in_speech = False
            silence_count = 0
            active_streak = 0
            frames_in_seg = 0
            voiced: List[bytes] = []

            while self.running:
                try:
                    frame = self.audio_q.get(timeout=0.5)
                except queue.Empty:
                    # also periodically flush pending if waiting too long
                    self._flush_old_pending()
                    continue

                if len(frame) != self.frame_bytes:
                    continue

                db = rms_dbfs(frame)
                energetic = (db >= self.energy_threshold_db)
                speech = energetic and self.vad.is_speech(frame, self.in_sr)

                if speech:
                    active_streak += 1
                    if not in_speech and active_streak >= self.min_active_frames:
                        in_speech = True
                        silence_count = 0
                        frames_in_seg = 0
                        voiced = []
                    if in_speech:
                        voiced.append(frame)
                        frames_in_seg += 1
                        if frames_in_seg >= self.max_segment_frames:
                            self._handle_segment(voiced)
                            in_speech = False
                            voiced = []
                            frames_in_seg = 0
                            silence_count = 0
                else:
                    active_streak = 0
                    if in_speech:
                        silence_count += 1
                        # protect very short speech from being cut
                        if frames_in_seg < self.min_segment_frames and silence_count < (self.silence_tail_frames * 2):
                            continue
                        if silence_count >= self.silence_tail_frames:
                            self._handle_segment(voiced)
                            in_speech = False
                            voiced = []
                            frames_in_seg = 0
                            silence_count = 0

                # Periodically try to reclassify/flush pending
                self._reclassify_pending_if_ready()
                self._flush_old_pending()

    def _handle_segment(self, frames: List[bytes]):
        if not frames:
            return
        pcm = b"".join(frames)
        a48 = pcm16_to_f32(pcm)
        a16 = resample(a48, self.in_sr, self.asr_sr)

        # Decode
        text = self.asr_decode(a16)
        if not text:
            return
        if looks_like_junk(text):
            return

        ts = time.time()
        # Provisional speaker
        spk = self._assign_speaker(a16, ts)
        # Queue as pending (to allow quick reclassification once both prototypes exist)
        self.pending.append({"ts": ts, "audio16": a16, "text": text, "spk": spk, "printed": False})

        # Immediate reclassify attempt + flush if queue too large/old
        self._reclassify_pending_if_ready()
        self._flush_old_pending()


# =========================
# ENTRY
# =========================
def main():
    app = RealtimeASRApp()
    try:
        blocksize = int(INPUT_SR * (FRAME_MS / 1000.0))
        print(f"[INFO] Opening input stream...", flush=True)
        app.run()
    except KeyboardInterrupt:
        try:
            # final flush pending + merge buffer
            while app.pending:
                item = app.pending.popleft()
                if not item.get("printed"):
                    app._merge_or_emit(item.get("spk") or "S1", item["text"])
            app._flush_merge_buf()
        except Exception:
            pass
        print("\nInterrupted by user.", flush=True)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()

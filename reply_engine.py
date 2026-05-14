#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reply_engine.py
Маршрутизация локальной LLM (Ollama) для телефонных подсказок.

Используем одну лёгкую модель для всего:
- FAST  (на партиалах S2): llama3.2:1b
- FINAL (на финалах S2):   llama3.2:1b

Исправлено:
- Жёсткий формат ответа модели: "S1=<реплика>" (ровно одна строка).
- Надёжная очистка ответа: убираем "Here's a possible reply:", "I'd be happy to help…",
  кавычки, префиксы "S1:", "Caller:", берём только саму фразу.
- Троттлинг FAST: ≥ 1 сек между вызовами.
"""

from __future__ import annotations
import os, json, asyncio, time, re
import urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ===================== КОНФИГ OLLAMA =====================

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# ОДНА модель для обоих путей
FAST_MODEL: str  = "llama3.2:1b"
FINAL_MODEL: str = "qwen2.5:7b-instruct"

# Настройки генерации
FAST_OPTS: dict = {
    "num_ctx": 2048,
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_predict": 18,   # ≤ ~20 слов
    "num_batch": 512
}
FINAL_OPTS: dict = {
    "num_ctx": 2048,
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_predict": 24,
    "num_batch": 128
}

# Ограничения и окна
MAX_HISTORY_TURNS: int = 8                 # сколько последних реплик держать (S1/S2 суммарно)
PARTIAL_MIN_WORDS: int = 2                 # не дёргать FAST, пока партиал слишком короткий
PARTIAL_DEBOUNCE_MS: int = 1700            # ← увеличено до 1 секунды
FAST_TIMEOUT_S: float = 1.2                # таймаут запросов FAST
FINAL_TIMEOUT_S: float = 2.5               # таймаут запросов FINAL

# Постобработка
WS_RE = re.compile(r"\s+")
QUOTE_CHARS = " \"'“”‘’`"

META_PREFIXES_RE = re.compile(
    r"(?i)^(?:here'?s a possible reply|possible reply|a possible reply|sample reply|"
    r"you can say|try|response|reply|answer|suggestion|suggested reply|"
    r"i'?d be happy to help.*?|i can help.*?|sure,.*?|certainly,.*?|"
    r"you could say|consider saying|for example|for instance)\s*:?\s*"
)

# ===================== ЧТЕНИЕ .env (на будущее) ======================

def _load_env(path: str = ".env") -> dict:
    cfg = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return cfg

_ENV = _load_env(".env")
OPEN_API_2: str = _ENV.get("OPEN_API_2", "").strip()
OPENAI_ORG_ID: str = _ENV.get("OPENAI_ORG_ID", "").strip()

# ===================== HTTP УТИЛИТЫ =====================

def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "ignore")
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "bad_json_response", "raw": raw[:400]}

async def _aio_post_json(url: str, payload: dict, timeout: float) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _http_post_json, url, payload, timeout)

# ===================== ПОСТОБРАБОТКА ТЕКСТА =====================

def _first_sentence(text: str) -> str:
    """Вернуть первую фразу по . ! ? — если есть."""
    for p in ('.', '!', '?'):
        idx = text.find(p)
        if idx != -1:
            return text[: idx + 1].strip()
    return text.strip()

def _strip_quotes(text: str) -> str:
    """Снять внешние кавычки."""
    t = text.strip()
    if len(t) >= 2 and t[0] in QUOTE_CHARS and t[-1] in QUOTE_CHARS:
        return t[1:-1].strip()
    return t.strip(QUOTE_CHARS).strip()

def _normalize_sentence(raw: str) -> str:
    """
    Жёстко вытягиваем реплику:
      1) Ищем формат S1=<...> или S1: <...> или FINAL: <...>
      2) Если нет — берём текст в первых кавычках
      3) Сносим мета-префиксы ("Here's a possible reply:", и т.п.)
      4) Возвращаем первую фразу (ограничим позже по словам)
    """
    if not raw:
        return ""

    t = WS_RE.sub(" ", raw).strip()

    # 1) Форматные маркеры
    m = re.search(r"(?i)\bS1\s*[=:]\s*(.+)", t)
    if not m:
        m = re.search(r"(?i)\bFINAL\s*:\s*(.+)", t)
    if m:
        t = m.group(1).strip()
    else:
        # 2) Попробовать взять содержимое в кавычках
        qm = re.search(r"[\"“‘']\s*(.+?)\s*[\"”’']", t)
        if qm:
            t = qm.group(1).strip()
        else:
            # 3) Снести типовые мета-префиксы типа "Here's a possible reply:"
            t = META_PREFIXES_RE.sub("", t)

    # Снять внешние кавычки/мусор, префиксы "S1:" если остались
    t = _strip_quotes(t)
    t_low = t.lower()
    for pref in ("s1:", "caller:", "you:"):
        if t_low.startswith(pref):
            t = t[len(pref):].lstrip()
            break

    # Удалить "—", ":" в конце
    if t.endswith(":") or t.endswith("—"):
        t = t[:-1].rstrip()

    # Первая фраза
    t = _first_sentence(t)
    return t

def _cut_to_words(text: str, max_words: int = 20) -> str:
    words = [w for w in WS_RE.sub(" ", text).split(" ") if w]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])

# ===================== ПРОМПТЫ =====================

def build_prompt_fast(goal: str, context: str, history: List[Tuple[str,str]], s2_partial: str) -> str:
    """
    Жёсткий формат: вернуть ОДНУ строку строго в формате
      S1=<ваше короткое предложение без кавычек>
    """
    hist_lines = [f"{lab}: {txt}" for (lab, txt) in history[-MAX_HISTORY_TURNS:]]
    hist_text = "\n".join(hist_lines) if hist_lines else "(no recent turns)"
    s2_part = _cut_to_words(s2_partial, 40)
    return (
        "You draft the caller's next line (S1) in a live UK-English phone call. "
        "Other party is S2 (agent). Produce ONE short, natural, goal-oriented reply (≤ 20 words).\n"
        "Output format: exactly one line => S1=<sentence>\n"
        "Do NOT include any other text, comments, quotes, code blocks, or explanations.\n\n"
        f"Goal: {goal}\n"
        f"Context: {context}\n\n"
        f"Recent transcript:\n{hist_text}\n"
        f"S2 (partial): {s2_part}\n\n"
        "S1="
    )

def build_prompt_final(goal: str, context: str, history: List[Tuple[str,str]]) -> str:
    """
    Жёсткий формат: вернуть ОДНУ строку строго в формате
      S1=<ваше короткое предложение без кавычек>
    """
    hist_lines = [f"{lab}: {txt}" for (lab, txt) in history[-MAX_HISTORY_TURNS:]]
    hist_text = "\n".join(hist_lines) if hist_lines else "(no recent turns)"
    return (
        "You speak for the caller (S1) in a UK-English phone call. "
        "Given the recent transcript, produce ONE concise, polite, goal-driven reply (≤ 20 words).\n"
        "Output format: exactly one line => S1=<sentence>\n"
        "Do NOT include any other text, comments, quotes, code blocks, or explanations.\n\n"
        f"Goal: {goal}\n"
        f"Context: {context}\n\n"
        f"Recent transcript:\n{hist_text}\n\n"
        "S1="
    )

# ===================== ОСНОВНОЙ ДВИЖОК =====================

@dataclass
class ReplyEngine:
    goal: str
    context: str
    history: List[Tuple[str,str]] = field(default_factory=list)
    _last_fast_ms: float = 0.0

    def add_history(self, label: str, text: str):
        t = (label, (text or "").strip())
        if not t[1]:
            return
        self.history.append(t)
        if len(self.history) > (MAX_HISTORY_TURNS * 2):
            self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

    async def fast_reply_from_partial(self, s2_partial: str) -> Optional[str]:
        if not s2_partial or len(s2_partial.strip().split()) < PARTIAL_MIN_WORDS:
            return None
        now_ms = time.monotonic() * 1000.0
        if (now_ms - self._last_fast_ms) < PARTIAL_DEBOUNCE_MS:
            return None
        self._last_fast_ms = now_ms

        prompt = build_prompt_fast(self.goal, self.context, self.history, s2_partial)
        url = f"{OLLAMA_HOST}/api/generate"
        payload = {"model": FAST_MODEL, "prompt": prompt, "stream": False, "options": FAST_OPTS}
        try:
            resp = await _aio_post_json(url, payload, timeout=FAST_TIMEOUT_S)
            if "error" in resp:
                return None
            out = _normalize_sentence(resp.get("response", ""))
            return _cut_to_words(out, 20) if out else None
        except Exception:
            return None

    async def final_reply(self) -> Optional[str]:
        prompt = build_prompt_final(self.goal, self.context, self.history)
        url = f"{OLLAMA_HOST}/api/generate"
        payload = {"model": FINAL_MODEL, "prompt": prompt, "stream": False, "options": FINAL_OPTS}
        try:
            resp = await _aio_post_json(url, payload, timeout=FINAL_TIMEOUT_S)
            if "error" in resp:
                return None
            out = _normalize_sentence(resp.get("response", ""))
            return _cut_to_words(out, 20) if out else None
        except Exception:
            return None

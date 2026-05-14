#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark fast non-reasoning OpenAI models (GPT-5-ready) via Responses API.

• Берёт ключи из .env: OPENAI_API_KEY или OPENAI_API; опц. OPENAI_ORG_ID
• Одинаковый диалог-промпт: модель должна ответить за S1 одной короткой фразой
• Для GPT-5* — messages+content[type='input_text'], без temperature, с text.verbosity='low' и reasoning.effort='minimal'
• Для 4.1/4o — классические поля temperature / max_output_tokens
• Надёжный парсинг ответа: если output_text пуст, разбираем response.output[*].content[*].text
• Авто-удаление неподдерживаемых параметров при 400 "Unsupported parameter"
• Подробная диагностика: токены, tok/s, типы блоков output

Пример запуска:
  pip install --upgrade openai
  python test.py --models gpt-5-mini,gpt-5-nano,gpt-4o-mini,gpt-4.1-mini,gpt-4.1-nano --runs 1 --verbose
"""

import os, sys, json, time, argparse
from statistics import mean
from typing import List, Dict, Tuple, Optional

try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APIStatusError,
        AuthenticationError,
        NotFoundError,
        RateLimitError,
        OpenAIError,
    )
except Exception:
    sys.stderr.write(
        "[ERROR] Не найден пакет 'openai'. Установите:\n"
        "  pip install --upgrade openai\n"
    )
    raise

DEFAULT_MODELS = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

SAMPLE_DIALOG_EN = """\
Context: UK customer calling electricity supplier to arrange a smart meter installation. The caller is a polite 35-year-old man named Andrew. Be concise and goal-oriented.

Conversation so far (S1 = caller, S2 = agent):
S1: Hello, I’m calling to arrange a smart meter installation for my flat in Blackpool.
S2: Hello, certainly. Could I take your postcode, please?
S1: It’s FY2 0AB.
S2: Thank you. Are you the account holder?
S1: Yes, my name is Andrew; the account is under my name.
S2: Great. Do you have easy access to the meter location?
S1: Yes, it’s in the hallway cupboard, easy to reach.
S2: Perfect. We have appointments next Tuesday morning or Thursday afternoon. Which suits you?
S1: Tuesday morning would be ideal if that’s available.
S2: We have 09:30 or 11:45 on Tuesday. Which time would you like?
S1: 09:30 sounds good, but I’d like to confirm the installation details first.

Task for the model:
You are S1 (the caller). Reply with ONE short sentence (≤ 20 words) that politely moves us closer to confirming a smart-meter installation at the earliest suitable time. Use UK English. Output ONLY the sentence, nothing else.
"""

VERIFY_HINT = (
    "ORG_VERIFY_REQUIRED: модель требует верификацию организации.\n"
    "• Settings → Organization → General → Verify Organization\n"
    "• После подтверждения: подождите до 30 мин, обновите сессию, используйте Project API-key и корректный org id.\n"
)

# ---------- utils ----------

def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))
    except Exception as e:
        sys.stderr.write(f"[WARN] Не удалось прочитать {path}: {e}\n")

def mask_key(k: str) -> str:
    if not k: return "-"
    return (k[:6] + "…" + k[-4:]) if len(k) > 12 else (k[:2] + "…" + k[-2:])

def build_client() -> Tuple[OpenAI, str]:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API")
    if not api_key:
        sys.stderr.write(
            "[FATAL] Нет ключа OPENAI_API_KEY (или OPENAI_API) в окружении/.env\n"
            "Пример .env:\nOPENAI_API_KEY=sk-...\nOPENAI_ORG_ID=org_...   # опционально\n"
        )
        sys.exit(2)
    org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    try:
        client = OpenAI(api_key=api_key, organization=org) if org else OpenAI(api_key=api_key)
    except Exception as e:
        sys.stderr.write(f"[FATAL] Не удалось инициализировать OpenAI клиент: {e}\n")
        sys.exit(2)
    print(f"[INFO] Using API key: {mask_key(api_key)}")
    print(f"[INFO] Organization header: {org if org else '(auto from key)'}")
    return client, (org or "")

def visible_models(client: OpenAI) -> Optional[set]:
    try:
        resp = client.models.list()
        return {m.id for m in (getattr(resp, "data", None) or [])}
    except OpenAIError as e:
        sys.stderr.write(f"[WARN] /v1/models недоступен: {e}\n")
        return None

def classify_error(e: Exception) -> Tuple[str, str]:
    s = (getattr(e, "message", None) or str(e) or "").lower()
    code = getattr(e, "status_code", None)
    if isinstance(e, APIStatusError):
        if code == 400 and ("must be verified" in s or "verify organization" in s):
            return "ORG_VERIFY_REQUIRED", VERIFY_HINT.strip()
        if code == 400 and "unsupported parameter" in s:
            return "UNSUPPORTED_PARAM", str(e)
        if code == 401:
            return "AUTH", "AUTH: неверный/отозванный ключ или ключ не принадлежит этой организации/проекту."
        if code == 404 or "not found" in s:
            return "MODEL_NOT_FOUND", "MODEL_NOT_FOUND: модель не найдена/недоступна."
        if code == 429 or "rate limit" in s:
            return "RATE_LIMIT", "RATE_LIMIT: превышена квота/частота."
        return f"API_{code or 'ERR'}", f"API error {code or ''}: {getattr(e, 'message', e)}"
    if isinstance(e, AuthenticationError):
        return "AUTH", "AUTH: проверьте OPENAI_API_KEY и (опц.) OPENAI_ORG_ID."
    if isinstance(e, NotFoundError):
        return "MODEL_NOT_FOUND", "MODEL_NOT_FOUND: модель не найдена/недоступна."
    if isinstance(e, RateLimitError):
        return "RATE_LIMIT", "RATE_LIMIT: превышена квота/частота."
    if isinstance(e, APIConnectionError):
        return "CONNECTION", f"CONNECTION: сетевой сбой {e}"
    if isinstance(e, OpenAIError):
        return "OPENAI_ERR", f"OpenAIError: {e}"
    return "ERROR", str(e)

def is_gpt5(model: str) -> bool:
    return model.startswith("gpt-5")

def build_messages(prompt: str):
    """Корректный формат для Responses API: type='input_text'."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt}
            ],
        }
    ]

def extract_text_and_kinds(resp) -> Tuple[str, str]:
    """Надёжный извлекатель текста + перечень типов блоков в output."""
    kinds = []
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip(), "output_text"
    try:
        out = getattr(resp, "output", None) or []
        parts = []
        for item in out:
            k = getattr(item, "type", None) or getattr(item, "role", None) or item.__class__.__name__
            if k: kinds.append(str(k))
            content = getattr(item, "content", None) or []
            for c in content:
                t = getattr(c, "text", None)
                if t: parts.append(t)
        return ("".join(parts).strip(), "+".join(sorted(set(kinds))) if kinds else "")
    except Exception:
        return "", ""

# ---------- core call ----------

def call_once(client: OpenAI, model: str, prompt: str, attempt: int = 0):
    """
    Один нестриминговый вызов с параметрами, совместимыми с моделью.
    GPT-5*: messages + input_text; без temperature; быстрые пресеты text/reasoning.
    4.1/4o: классические temperature/max_output_tokens.
    При 400 Unsupported parameter — убираем поле и повторяем.
    """
    params = {
        "model": model,
        "input": build_messages(prompt),
        "user": "model_bench_s1_reply",
        "metadata": {"app": "model_bench_s1_reply", "attempt": str(attempt)},
    }
    if is_gpt5(model):
        params["text"] = {"verbosity": "low"}          # быстрый краткий текст
        params["reasoning"] = {"effort": "minimal"}    # минимальные рассуждения
        # без temperature/max_output_tokens — часто не поддерживаются в 5-й серии
    else:
        params["temperature"] = 0.2
        params["max_output_tokens"] = 32

    removed = set()
    while True:
        t0 = time.perf_counter()
        try:
            resp = client.responses.create(**params)
            dt = time.perf_counter() - t0
            return resp, dt
        except APIStatusError as e:
            msg = (getattr(e, "message", "") or str(e)).lower()
            # Автоматическое удаление неподдерживаемых полей
            if "unsupported parameter" in msg:
                changed = False
                for p in ("temperature", "max_output_tokens", "text", "reasoning", "modalities"):
                    if f"'{p}'" in msg and p in params and p not in removed:
                        removed.add(p)
                        params.pop(p, None)
                        changed = True
                if changed:
                    continue
            # Прочие ошибки — наружу
            raise

# ---------- runner / reporting ----------

def run_bench(client: OpenAI, models: List[str], prompt: str, runs: int) -> List[Dict]:
    results = []
    for m in models:
        print("\n" + "=" * 80)
        print(f"[MODEL] {m}")
        times: List[float] = []
        last = None
        status, err = "OK", ""
        try:
            for i in range(runs):
                try:
                    resp, dt = call_once(client, m, prompt, attempt=i)
                    times.append(dt)
                    last = resp
                except (APIConnectionError, RateLimitError) as e:
                    status, err = classify_error(e)
                    print(f"[WARN] {status}: {err} — попытка {i+1}/{runs} пропущена")
                    time.sleep(min(2 ** i, 4))
        except Exception as e:
            status, err = classify_error(e)

        summary = {
            "model": m,
            "status": status,
            "error": err,
            "runs": len(times),
            "latency_ms_avg": round(mean(times) * 1000, 1) if times else None,
            "latency_ms_each": [round(x * 1000, 1) for x in times],
            "output_text": None,
            "output_kinds": "",
            "usage": None,
        }

        if last is not None:
            text, kinds = extract_text_and_kinds(last)
            summary["output_text"] = text
            summary["output_kinds"] = kinds
            use = getattr(last, "usage", None)
            if use and times:
                info = {
                    "input_tokens": getattr(use, "input_tokens", None),
                    "output_tokens": getattr(use, "output_tokens", None),
                    "total_tokens": getattr(use, "total_tokens", None),
                }
                try:
                    out_tok = info["output_tokens"] or 0
                    avg_sec = mean(times)
                    info["speed_tok_per_s"] = round(out_tok / avg_sec, 1) if avg_sec > 0 else None
                except Exception:
                    pass
                summary["usage"] = info

        if summary["status"] == "OK" and summary["runs"] > 0:
            print(f"[OK]  avg_latency = {summary['latency_ms_avg']} ms  (runs={summary['runs']}, each={summary['latency_ms_each']})")
            if summary["usage"]:
                ut = summary["usage"]
                print(f"[TOKENS] in={ut.get('input_tokens')} out={ut.get('output_tokens')} total={ut.get('total_tokens')}  "
                      f"speed≈{ut.get('speed_tok_per_s')} tok/s")
            print("[REPLY] " + (summary["output_text"] or "<empty>"))
            if summary["output_kinds"]:
                print(f"[OUTPUT_KINDS] {summary['output_kinds']}")
        else:
            print(f"[ERR] {summary['status']} — {summary['error'] or 'см. детали выше'}")

        results.append(summary)
    return results

def print_summary(results: List[Dict]) -> None:
    ok = [r for r in results if r["status"] == "OK" and r["latency_ms_avg"] is not None]
    ko = [r for r in results if r["status"] != "OK" or r["latency_ms_avg"] is None]
    ok.sort(key=lambda r: r["latency_ms_avg"])

    print("\n" + "#" * 80)
    print("# SUMMARY (быстрее сверху)")
    print("#" * 80)

    def row(r: Dict) -> str:
        ut = r.get("usage") or {}
        return (
            f"{r['model']:<18} | {r['status']:<18} | "
            f"{(str(r['latency_ms_avg']) + ' ms') if r['latency_ms_avg'] is not None else '-':>10} | "
            f"in:{ut.get('input_tokens','-'):>5}  out:{ut.get('output_tokens','-'):>5}  tot:{ut.get('total_tokens','-'):>5} | "
            f"speed:{(str(ut.get('speed_tok_per_s'))+' tok/s') if ut.get('speed_tok_per_s') else '-':>10}"
        )

    if ok:
        print("\n".join(row(r) for r in ok))
    if ko:
        if ok: print("-" * 80)
        print("\n".join(row(r) + (f"  :: {r.get('error','')}" if r.get('error') else "") for r in ko))

def main():
    parser = argparse.ArgumentParser(description="Benchmark fast OpenAI models (GPT-5-ready).")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="Список моделей через запятую")
    parser.add_argument("--runs", type=int, default=1, help="Повторов на модель")
    parser.add_argument("--prompt-file", type=str, default="", help="Файл с диалогом (если хотите заменить встроенный)")
    parser.add_argument("--verbose", action="store_true", help="Печатать список видимых моделей")
    args = parser.parse_args()

    client, _ = build_client()

    vis = visible_models(client)
    if vis is not None:
        print(f"[INFO] Видимых моделей: {len(vis)}")
        # if args.verbose:
        #     print("[INFO] " + ", ".join(sorted(vis)))
    else:
        print("[INFO] /v1/models недоступен (продолжаем).")

    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
        except Exception as e:
            sys.stderr.write(f"[FATAL] Не удалось прочитать {args.prompt_file}: {e}\n")
            sys.exit(3)
    else:
        prompt = SAMPLE_DIALOG_EN

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print("[INFO] Тестируем модели: " + ", ".join(models))
    if vis is not None:
        marks = ["✓" if m in vis else "•" for m in models]
        print("[INFO] Доступность по /v1/models: " + "  ".join(f"{mk} {m}" for mk, m in zip(marks, models)))

    results = run_bench(client, models, prompt, args.runs)
    print_summary(results)

    print("\n" + "#" * 80)
    print("# REPLIES")
    print("#" * 80)
    for r in results:
        reply = r.get("output_text") or ""
        preview = (reply[:280] + "…") if len(reply) > 280 else reply
        print(f"{r['model']:<18} | {r['status']:<18} | {preview}")

    print("\n[DONE] Бенч завершён. GPT-5 теперь вызывается корректно (input_text), ответы извлекаются устойчиво.\n")

if __name__ == "__main__":
    main()

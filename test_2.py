#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama bench for short S1 replies (≤ 20 words):
- Замеряет TTFT (первый токен), полное время, токены, tok/s.
- Работает через /api/generate (stream), без сторонних зависимостей.
- Поддерживает прогрев (--warmup), повторения (--runs), драфт-модель (speculative, --draft-for).
- По умолчанию тестирует ваши локальные модели; легко добавить qwen2.5:7b-instruct/3b-instruct.

Примеры:
  python ollama_bench_s1.py
  python ollama_bench_s1.py --models "llama3.2:1b,mistral:latest,qwen2.5:14b-instruct,qwen2.5:7b-instruct" --runs 5 --warmup 1
  python ollama_bench_s1.py --draft-for "qwen2.5:14b-instruct=llama3.2:1b"
  OLLAMA_HOST=http://localhost:11434 python ollama_bench_s1.py
"""

import os
import sys
import time
import json
import math
import argparse
import statistics
import requests
from typing import Dict, List, Optional, Tuple

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODELS = [
    # ваши текущие модели (без 20B reasoning)
    "llama3.2:1b",
    "mistral:latest",
    "qwen2.5:14b-instruct",
    # добавите qwen2.5:7b-instruct после pull — просто добавьте в --models
]

# Короткий диалог и задача (как в ваших тестах)
PROMPT = """Context: UK customer calling electricity provider to arrange a smart meter installation. Caller: polite 35-year-old man named Andrew. Be concise and goal-oriented.

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

Task:
You are S1 (the caller). Reply with ONE short sentence (≤ 20 words) that moves us closer to confirming the earliest smart-meter installation. Use UK English. Output ONLY the sentence.
"""

def parse_draft_map(raw: str) -> Dict[str, str]:
    """
    "--draft-for 'bigA=smallX,bigB=smallX'" -> {"bigA":"smallX","bigB":"smallX"}
    """
    if not raw:
        return {}
    m = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            m[k] = v
    return m

def default_options_for_model(model: str, max_tokens: int = 24) -> Dict:
    """
    Базовые быстрые опции (правим по размеру модели).
    num_ctx: достаточно для короткого промпта; num_batch: выше — быстрее (если хватает VRAM).
    """
    opts = {
        "num_ctx": 2048,
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
        "num_predict": max_tokens,  # ограничиваем ответ
    }
    low = model.lower()
    # Грубая эвристика под batch (регулируйте вручную при нехватке VRAM):
    if "14b" in low or ":14b" in low:
        opts["num_batch"] = 128
    elif "7b" in low or ":7b" in low or "mistral" in low:
        opts["num_batch"] = 256
    else:
        # 3b/1b
        opts["num_batch"] = 512
    return opts

def generate_once(host: str, model: str, prompt: str, options: Dict, draft: Optional[str]) -> Tuple[Dict, str]:
    """
    Один запуск /api/generate со стримингом. Возвращает (метрики, ответ).
    Метрики: ttft_ms, total_ms, load_ms, prompt_eval_ms, eval_ms, prompt_tokens, gen_tokens, tok_s
    """
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": dict(options),
    }
    if draft:
        # speculative decoding с маленькой моделью, ускоряет большие
        payload["options"]["draft"] = draft

    # Засекаем
    t_start = time.perf_counter()
    ttft = None
    answer_parts: List[str] = []

    # Для конечной сводки берём числа из завершающего чанка.
    final_eval_count = None
    final_prompt_eval_count = None
    final_total_duration_ns = None
    final_load_duration_ns = None
    final_prompt_eval_duration_ns = None
    final_eval_duration_ns = None

    with requests.post(url, json=payload, stream=True, timeout=60) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(delimiter=b"\n"):
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line.decode("utf-8"))
            except Exception:
                # Игнорируем битые строки
                continue

            # В обычных стрим-чанках приходит поле "response" (кусочек текста)
            if "response" in obj:
                if ttft is None and obj.get("response"):
                    ttft = (time.perf_counter() - t_start) * 1000.0  # мс
                answer_parts.append(obj.get("response", ""))

            # В финальном чанке: done=true + метаданные по таймингам/токенам
            if obj.get("done"):
                final_eval_count = obj.get("eval_count")
                final_prompt_eval_count = obj.get("prompt_eval_count")
                final_total_duration_ns = obj.get("total_duration")
                final_load_duration_ns = obj.get("load_duration")
                final_prompt_eval_duration_ns = obj.get("prompt_eval_duration")
                final_eval_duration_ns = obj.get("eval_duration")
                break

    t_total_ms = (time.perf_counter() - t_start) * 1000.0
    if ttft is None:
        # Не получили ни одного чанка текста — считаем TTFT=total (редко, но пусть будет)
        ttft = t_total_ms

    # Переводим наносекунды в мс
    def ns_to_ms(x):
        return (x / 1e6) if isinstance(x, (int, float)) and x is not None else None

    load_ms = ns_to_ms(final_load_duration_ns)
    prompt_eval_ms = ns_to_ms(final_prompt_eval_duration_ns)
    eval_ms = ns_to_ms(final_eval_duration_ns)

    prompt_tokens = int(final_prompt_eval_count or 0)
    gen_tokens = int(final_eval_count or 0)

    tok_s = None
    if eval_ms and eval_ms > 0 and gen_tokens > 0:
        tok_s = round(gen_tokens / (eval_ms / 1000.0), 2)

    metrics = {
        "ttft_ms": round(ttft, 1),
        "total_ms": round(t_total_ms, 1),
        "load_ms": round(load_ms, 1) if load_ms is not None else None,
        "prompt_eval_ms": round(prompt_eval_ms, 1) if prompt_eval_ms is not None else None,
        "eval_ms": round(eval_ms, 1) if eval_ms is not None else None,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "tok_s": tok_s,
    }
    answer = "".join(answer_parts).strip()
    return metrics, answer

def p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    k = max(0, min(len(vs)-1, int(math.ceil(0.95*len(vs))-1)))
    return vs[k]

def run_for_model(host: str, model: str, prompt: str, runs: int, warmup: int,
                  options: Dict, draft_map: Dict[str, str]) -> Dict:
    draft = draft_map.get(model)
    # прогрев
    for _ in range(max(0, warmup)):
        try:
            generate_once(host, model, "warmup", options, draft)
        except Exception:
            pass

    # боевые прогоны
    latencies_total = []
    latencies_ttft = []
    answers = []
    last_metrics = None
    for i in range(runs):
        try:
            m, ans = generate_once(host, model, prompt, options, draft)
            latencies_total.append(m["total_ms"])
            latencies_ttft.append(m["ttft_ms"])
            answers.append(ans)
            last_metrics = m
        except Exception as e:
            answers.append(f"<ERROR: {e}>")

    result = {
        "model": model,
        "runs": runs,
        "opts": options,
        "draft": draft,
        "ttft_ms_each": latencies_ttft,
        "total_ms_each": latencies_total,
        "ttft_ms_avg": round(statistics.mean(latencies_ttft), 1) if latencies_ttft else None,
        "ttft_ms_med": round(statistics.median(latencies_ttft), 1) if latencies_ttft else None,
        "total_ms_avg": round(statistics.mean(latencies_total), 1) if latencies_total else None,
        "total_ms_med": round(statistics.median(latencies_total), 1) if latencies_total else None,
        "total_ms_p95": round(p95(latencies_total), 1) if latencies_total else None,
        "answers": answers,
        "last_metrics": last_metrics,
    }
    return result

def print_model_result(res: Dict):
    print("\n" + "="*90)
    print(f"[MODEL] {res['model']}")
    if res.get("draft"):
        print(f"[DRAFT] {res['draft']}")
    print(f"[OPTS ] {json.dumps(res['opts'], ensure_ascii=False)}")
    print(f"[RUNS ] {res['runs']}")
    print(f"[TTFT ] median={res['ttft_ms_med']} ms  avg={res['ttft_ms_avg']} ms  each={res['ttft_ms_each']}")
    print(f"[TOTAL] median={res['total_ms_med']} ms  avg={res['total_ms_avg']} ms  p95={res['total_ms_p95']} ms  each={res['total_ms_each']}")
    m = res.get("last_metrics") or {}
    if m:
        print(f"[TOKENS] prompt={m.get('prompt_tokens')} gen={m.get('gen_tokens')}  tok/s≈{m.get('tok_s')}")
    # Покажем последний ответ (обычно достаточно 1)
    if res["answers"]:
        print(f"[REPLY] {res['answers'][-1]}")

def print_summary(allres: List[Dict], sort_key: str):
    print("\n" + "#"*90)
    print("# SUMMARY  (sort by: {})".format(sort_key))
    print("#"*90)
    header = f"{'model':<28} | {'med_total(ms)':>12} | {'avg_total(ms)':>12} | {'p95(ms)':>9} | {'med_TTFT(ms)':>12} | {'tok/s':>8}"
    print(header)
    print("-"*len(header))
    rows = []
    for r in allres:
        m = r.get("last_metrics") or {}
        rows.append({
            "model": r["model"] + (f" [draft={r['draft']}]" if r.get("draft") else ""),
            "med_total": r["total_ms_med"],
            "avg_total": r["total_ms_avg"],
            "p95": r["total_ms_p95"],
            "med_ttft": r["ttft_ms_med"],
            "toks": m.get("tok_s"),
        })
    rows.sort(key=lambda x: (float('inf') if x[sort_key is None] else x[sort_key]) if isinstance(sort_key, (int, float)) else x.get(sort_key.replace('_total','med_total'), float('inf')))
    # исправим сортировку по строковому ключу
    if isinstance(sort_key, str):
        keymap = {
            "med_total": "med_total",
            "avg_total": "avg_total",
            "p95": "p95",
            "med_ttft": "med_ttft",
        }
        k = keymap.get(sort_key, "med_total")
        rows.sort(key=lambda x: x[k] if x[k] is not None else float('inf'))

    for x in rows:
        print(f"{x['model']:<28} | {str(x['med_total']):>12} | {str(x['avg_total']):>12} | {str(x['p95']):>9} | {str(x['med_ttft']):>12} | {str(x['toks']):>8}")

def main():
    ap = argparse.ArgumentParser(description="Ollama short-reply latency benchmark")
    ap.add_argument("--host", default=DEFAULT_HOST, help="OLLAMA_HOST (default: %(default)s)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Список моделей через запятую")
    ap.add_argument("--runs", type=int, default=3, help="Повторов на модель (боевых)")
    ap.add_argument("--warmup", type=int, default=1, help="Прогревов на модель")
    ap.add_argument("--max-tokens", type=int, default=24, help="Максимум токенов в ответе")
    ap.add_argument("--sort-by", default="med_total", choices=["med_total","avg_total","p95","med_ttft"],
                    help="Критерий сортировки в Summary")
    ap.add_argument("--draft-for", default="", help="Маппинг драфтов: bigA=smallX,bigB=smallX")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    draft_map = parse_draft_map(args.draft_for)

    allres = []
    for model in models:
        opts = default_options_for_model(model, max_tokens=args.max_tokens)
        res = run_for_model(args.host, model, PROMPT, runs=args.runs, warmup=args.warmup,
                            options=opts, draft_map=draft_map)
        print_model_result(res)
        allres.append(res)

    print_summary(allres, sort_key=args.sort_by)
    print("\n[DONE] Bench complete.\n")

if __name__ == "__main__":
    main()

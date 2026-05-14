# VoicePlus

Realtime voice assistant prototype for phone calls and spoken conversations. VoicePlus captures two audio streams, sends low-latency transcription to Speechmatics, displays live S1/S2 transcript lines, and can generate short response suggestions through local Ollama models or optional OpenAI helpers.

## Features

- Dual-stream call transcription for speaker/customer sides.
- Speechmatics realtime WebSocket transcription.
- Terminal UI with live partials, final transcript history, pause state, and colored output.
- Echo suppression heuristics between S1 and S2 streams.
- Optional translation pipeline.
- Local LLM reply suggestions through `reply_engine.py` and Ollama.
- Optional OpenAI helper calls for translating a Russian goal into English and generating useful phrases.
- Per-call logs written to `call_logs/`.

## Files

- `main.py` is the current entry point.
- `reply_engine.py` contains local LLM prompt routing and cleanup logic.
- `vp.py` and `vp_2.py` are earlier application variants kept for reference.
- `test.py` and `test_2.py` are local benchmark/scratch scripts for model latency testing.

## Configuration

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Required for transcription:

```env
SPEECHMATICS_API=replace-with-speechmatics-api-key
SPEECHMATICS_WSS=wss://eu2.rt.speechmatics.com/v2/
```

Optional AI helpers:

```env
OPENAI_API=replace-with-openai-api-key
OLLAMA_HOST=http://127.0.0.1:11434
ENABLE_SUGGEST=1
```

API keys and call logs must stay local. Do not commit `.env`, logs, audio captures, or generated transcripts.

## Development

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

The app expects local audio routing/capture to be configured on the host. For local suggestion generation, run Ollama and pull the models referenced by `reply_engine.py`.

## Status

This is an experimental realtime assistant. The public version keeps runtime configuration external and focuses on the transcription, UI, and response-suggestion workflow.

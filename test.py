#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, asyncio, json, contextlib
import websockets

API = os.environ.get("SPEECHMATICS_API")
ENDPOINT = "wss://eu2.rt.speechmatics.com/v2/"
ENCODING = "pcm_s16le"

def usage():
    print("Usage: SPEECHMATICS_API=... python test.py /path/to/file.raw [rate]")
    sys.exit(1)

async def _connect():
    headers = [("Authorization", f"Bearer {API}")]
    try:
        return await websockets.connect(ENDPOINT, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(ENDPOINT, extra_headers=headers, max_size=None)

async def run():
    if not API: usage()
    if len(sys.argv) < 2: usage()
    path = sys.argv[1]
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 16000

    ws = await _connect()

    # Старт распознавания
    await ws.send(json.dumps({
        "message": "StartRecognition",
        "audio_format": {"type": "raw", "encoding": ENCODING, "sample_rate": rate},
        "transcription_config": {
            "language": "en",
            "output_locale": "en-GB",
            "enable_partials": True,
            "max_delay_mode": "flexible",
            "max_delay": 1.5,
            "audio_filtering_config": {"volume_threshold": 0},
            "operating_point": "enhanced"
        }
    }))

    async def reader():
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                print("[raw]", msg[:200]); continue
            print(data)

    rtask = asyncio.create_task(reader())

    # Кормим аудио бинарными кадрами (~100мс на кадр)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(3200)  # 16kHz * 2 bytes * 1 ch * 0.1s
            if not chunk:
                break
            await ws.send(chunk)
            await asyncio.sleep(0.10)

    # Завершение
    await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))
    await asyncio.sleep(2)
    with contextlib.suppress(Exception):
        await ws.close()
    rtask.cancel()

if __name__ == "__main__":
    asyncio.run(run())

"""模型层冒烟测试: whisper-npu + paraformer-cpu"""
import sys, os, wave
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from asr import create_model

def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data.astype(np.float32) / 32768.0, sr

WAV = os.path.join(BASE, "test_zh.wav")
if not os.path.exists(WAV):
    print(f"[SKIP] 缺少 {WAV}（未随仓库分发），无法做转写测试；请放入任意 16k 中文语音 wav")
    sys.exit(0)
audio, sr = read_wav(WAV)
if sr != 16000:
    n = int(len(audio) * 16000 / sr)
    audio = np.interp(np.linspace(0, 1, n, endpoint=False), np.linspace(0, 1, len(audio), endpoint=False), audio)
print(f"audio: {len(audio)/16000:.1f}s")

# whisper-npu
m = create_model("whisper-npu")
import time
t0 = time.time(); m.load(); dt = time.time() - t0
print(f"[whisper-npu] load {dt:.1f}s, loaded={m.loaded}")
t0 = time.time()
text = m.transcribe(audio)
print(f"[whisper-npu] ({time.time()-t0:.2f}s): {text}")

# paraformer-cpu
m2 = create_model("paraformer-cpu")
t0 = time.time(); m2.load(); dt2 = time.time() - t0
print(f"[paraformer-cpu] load {dt2:.2f}s")
t0 = time.time()
text2 = m2.transcribe(audio)
print(f"[paraformer-cpu] ({time.time()-t0:.2f}s): {text2}")

# 未知模型报错测试
try:
    create_model("xxx")
    print("[ERROR] 应该报错")
except ValueError as e:
    print(f"[OK] 未知模型拒绝: {e}")


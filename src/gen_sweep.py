"""Generate a log-sine sweep reference stimulus for mic domain-shift capture.
50 Hz -> 7.5 kHz over 9 s, 48 kHz mono, with 0.7 s lead/trail silence."""
import numpy as np, wave
fs = 48000
sil = np.zeros(int(0.7 * fs))
T = 9.0
t = np.linspace(0, T, int(T * fs), endpoint=False)
f0, f1 = 50.0, 7500.0
L = T / np.log(f1 / f0)
K = T * f0 / np.log(f1 / f0)
sweep = np.sin(2 * np.pi * K * (np.exp(t / L) - 1.0))
fade = int(0.05 * fs)
w = np.ones_like(sweep); w[:fade] = np.linspace(0, 1, fade); w[-fade:] = np.linspace(1, 0, fade)
sweep *= w
sig = np.concatenate([sil, 0.6 * sweep, sil]).astype(np.float32)
pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
with wave.open("reference.wav", "wb") as wv:
    wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(fs); wv.writeframes(pcm.tobytes())
print("reference.wav %.1fs @ %dHz" % (len(sig) / fs, fs))

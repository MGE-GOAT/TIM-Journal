"""Report duration, RMS, peak (and per-channel) of a wav — sanity that a mic captured signal."""
import sys, wave, numpy as np
p = sys.argv[1]
w = wave.open(p, "rb")
n, ch, fs, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
raw = w.readframes(n)
dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
a = np.frombuffer(raw, dtype=dt).astype(np.float64)
full = float(np.iinfo(dt).max)
a /= full
if ch > 1:
    a = a.reshape(-1, ch)
    for c in range(ch):
        x = a[:, c]
        print("  ch%d rms=%.5f peak=%.5f" % (c, np.sqrt(np.mean(x**2)), np.max(np.abs(x))))
    a = a.mean(1)
print("%s: %.2fs %dHz ch=%d rms=%.5f peak=%.5f" % (p, n / fs, fs, ch, np.sqrt(np.mean(a**2)), np.max(np.abs(a))))

"""Compare idle spectra: INMP441 fan-off vs fan-on (same mic) and the conventional
mic. If the fan contributes, fan-on shows added energy (broadband or tonal)."""
import numpy as np, wave
from scipy.signal import welch

def load(p):
    w = wave.open(p, "rb"); n, ch, fs, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(w.readframes(n), dtype=dt).astype(np.float64) / np.iinfo(dt).max
    if ch > 1: a = a.reshape(-1, ch)[:, 0]
    return a, fs

off, fo = load("inmp_fanoff.wav"); on, fn = load("inmp_fanon.wav"); pc, fp = load("pc_silence.wav")
fo_f, Po = welch(off, fs=fo, nperseg=2048); on_f, Pn = welch(on, fs=fn, nperseg=2048)
pc_ff, Ppc = welch(pc, fs=fp, nperseg=4096)
def dbf(P): return 10 * np.log10(np.maximum(P, 1e-20))
# broadband energy in bands
for lo, hi in [(60, 250), (250, 1000), (1000, 3000), (3000, 7000)]:
    mo = (fo_f >= lo) & (fo_f <= hi)
    e_off = 10 * np.log10(Po[mo].sum()); e_on = 10 * np.log10(Pn[mo].sum())
    print("band %5d-%5d Hz: INMP fan-off %.1f dB  fan-on %.1f dB  (delta %.2f)" % (lo, hi, e_off, e_on, e_on - e_off))
# peak tonal check: largest fan-on minus fan-off bin
d = dbf(Pn) - dbf(Po)
k = np.argmax(d)
print("max fan-on excess: %.1f dB at %.0f Hz" % (d[k], on_f[k]))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 3.4))
    plt.semilogx(fo_f[1:], dbf(Po)[1:], label="INMP441 fan-off (idle)")
    plt.semilogx(on_f[1:], dbf(Pn)[1:], label="INMP441 fan-on (CPU 64C)")
    plt.semilogx(pc_ff[1:], dbf(Ppc)[1:], "--", alpha=.7, label="Conventional mic")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (dBFS)"); plt.xlim(50, 8000)
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=.3); plt.tight_layout()
    plt.savefig("silence_spec.png", dpi=140); print("PNG saved")
except Exception as e:
    print("no plot", e)

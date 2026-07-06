"""Domain-shift analysis: from the two captures of the SAME played sweep, estimate
each mic's magnitude frequency response via Welch PSD ratio |H|=sqrt(Pyy/Pxx)
(the sweep's own spectral shape cancels), smooth in 1/6-octave, normalize, and
emit a common log-frequency grid + a PNG. Also reports each mic's silence noise floor."""
import numpy as np, wave
from math import gcd
from scipy.signal import welch, resample_poly, correlate

def load(p):
    w = wave.open(p, "rb"); n, ch, fs, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(w.readframes(n), dtype=dt).astype(np.float64) / np.iinfo(dt).max
    if ch > 1: a = a.reshape(-1, ch)
    return a, fs

ref, fsr = load("reference.wav")
pc, fspc = load("pc_capture.wav")
inmp, fsi = load("inmp_capture.wav")
if inmp.ndim > 1: inmp = inmp[:, 0]      # INMP441 is on the left channel
if pc.ndim > 1: pc = pc[:, 0]
if ref.ndim > 1: ref = ref[:, 0]

def response(cap, fs_cap, ref48, fsr=48000, fmax=7000):
    if fs_cap != fsr:
        g = gcd(fs_cap, fsr); refc = resample_poly(ref48, fs_cap // g, fsr // g)
    else:
        refc = ref48.copy()
    xc = correlate(cap, refc, mode="full")
    lag = np.argmax(np.abs(xc)) - (len(refc) - 1)
    start = max(0, lag); seg = cap[start:start + len(refc)]
    if len(seg) < len(refc): seg = np.pad(seg, (0, len(refc) - len(seg)))
    nper = 4096 if fs_cap >= 44000 else 2048
    f, Pxx = welch(refc, fs=fs_cap, nperseg=nper)
    _, Pyy = welch(seg, fs=fs_cap, nperseg=nper)
    H = np.sqrt(np.maximum(Pyy, 1e-20) / np.maximum(Pxx, 1e-20))
    m = (f >= 55) & (f <= fmax)
    return f[m], 20 * np.log10(H[m])

def smooth_oct(f, H, frac=6):
    out = np.empty_like(H)
    for i, fc in enumerate(f):
        lo, hi = fc / 2 ** (1 / (2 * frac)), fc * 2 ** (1 / (2 * frac))
        sel = (f >= lo) & (f <= hi)
        out[i] = H[sel].mean() if sel.any() else H[i]
    return out

def norm_mid(f, H):
    return H - H[(f >= 200) & (f <= 500)].mean()

fpc, Hpc = response(pc, fspc, ref); fin, Hin = response(inmp, fsi, ref)
Hpc = norm_mid(fpc, smooth_oct(fpc, Hpc, 3)); Hin = norm_mid(fin, smooth_oct(fin, Hin, 3))
grid = np.geomspace(60, 7000, 32)
pc_g = np.interp(grid, fpc, Hpc); in_g = np.interp(grid, fin, Hin)

# noise floor from the trailing 0.4 s of each capture (clean silence after the sweep)
nf_pc = 20 * np.log10(np.sqrt(np.mean(pc[-int(0.4 * fspc):] ** 2)) + 1e-12)
nf_in = 20 * np.log10(np.sqrt(np.mean(inmp[-int(0.4 * fsi):] ** 2)) + 1e-12)
print("# noise_floor_dBFS  PC=%.1f  INMP441=%.1f  (delta %.1f dB)" % (nf_pc, nf_in, nf_in - nf_pc))
print("# freq_Hz  PC_dB  INMP441_dB")
for fq, a, b in zip(grid, pc_g, in_g):
    print("%.1f %.2f %.2f" % (fq, a, b))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.2))
    plt.semilogx(grid, pc_g, "-o", ms=3, label="Conventional mic (training)")
    plt.semilogx(grid, in_g, "-s", ms=3, label="INMP441 (deployment)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Relative response (dB)")
    plt.legend(); plt.grid(True, which="both", alpha=.3); plt.tight_layout()
    plt.savefig("domainshift.png", dpi=140); print("# PNG saved")
except Exception as e:
    print("# no plot:", e)

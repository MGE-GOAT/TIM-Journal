"""Fan-noise domain-shift: for each mic, idle (fan-only) noise floor relative to
the played sweep -> a sensitivity-independent noise-to-signal ratio comparable
across mics. The INMP441 sits under the Pi fan, so its idle/sweep ratio should be
far higher (louder persistent background) -> the settled background PCEN normalizes."""
import numpy as np, wave
from scipy.signal import welch

def load(p):
    w = wave.open(p, "rb"); n, ch, fs, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(w.readframes(n), dtype=dt).astype(np.float64) / np.iinfo(dt).max
    if ch > 1: a = a.reshape(-1, ch)
    return a, fs

pc, fspc = load("pc_capture.wav"); inmp, fsi = load("inmp_capture.wav")
if pc.ndim > 1: pc = pc[:, 0]
if inmp.ndim > 1: inmp = inmp[:, 0]

def nsr(cap, fs, fmax=7000):
    sweep = cap[int(2.0 * fs):int(10.5 * fs)]   # played sweep window
    sil = cap[-int(0.5 * fs):]                   # trailing idle (fan only)
    nper = 4096 if fs >= 44000 else 2048
    f, Ps = welch(sweep, fs=fs, nperseg=nper)
    _, Pn = welch(sil, fs=fs, nperseg=min(nper, len(sil) // 2))
    m = (f >= 60) & (f <= fmax)
    return f[m], 10 * np.log10(np.maximum(Pn[m], 1e-20) / np.maximum(Ps[m], 1e-20))

fp, npc = nsr(pc, fspc); fi, nin = nsr(inmp, fsi)
grid = np.geomspace(60, 7000, 32)
pg = np.interp(grid, fp, npc); ig = np.interp(grid, fi, nin)
print("# idle-noise-to-sweep ratio (dB); higher = louder persistent background")
print("# freq  PC_dB  INMP441_dB")
for f, a, b in zip(grid, pg, ig):
    print("%.1f %.1f %.1f" % (f, a, b))
print("# broadband: PC=%.1f dB  INMP441=%.1f dB  (INMP is %.1f dB louder rel. to signal)"
      % (npc.mean(), nin.mean(), nin.mean() - npc.mean()))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.2))
    plt.semilogx(grid, pg, "-o", ms=3, label="Conventional mic (training)")
    plt.semilogx(grid, ig, "-s", ms=3, label="INMP441 under Pi fan (deployment)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Idle noise / sweep (dB)")
    plt.legend(); plt.grid(True, which="both", alpha=.3); plt.tight_layout()
    plt.savefig("fan.png", dpi=140); print("# PNG saved")
except Exception as e:
    print("# no plot", e)

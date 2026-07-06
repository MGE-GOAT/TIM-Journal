"""build_twin_filter.py -- the conventional->INMP441 'digital microphone twin'.
Both mics recorded the SAME played sweep, so the ratio of their captured PSDs IS
the mic-to-mic transfer H = |INMP441| / |conventional| (the played sweep cancels).
Build a 16 kHz FIR from |H(f)|, save it, and validate that filtering a conventional
clip reproduces the measured 0.5-2.4 kHz emphasis (Fig. 1)."""
import os, numpy as np, wave, json
from scipy.signal import welch, resample_poly, firwin2, freqz, lfilter

def load(p):
    w = wave.open(p, "rb"); n, ch, fs, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(w.readframes(n), dtype=dt).astype(np.float64) / np.iinfo(dt).max
    if ch > 1: a = a.reshape(-1, ch)[:, 0]
    return a, fs

FS = 16000
pc, fspc = load("pc_capture.wav")      # 48 kHz conventional (GM303)
inm, fsi = load("inmp_capture.wav")    # 16 kHz INMP441 (ch0)
# bring conventional to 16 kHz
pc16 = resample_poly(pc, FS, fspc)
# sweep window (played ~1.9-10.9 s into the 13 s capture)
sl = slice(int(2.0 * FS), int(10.5 * FS))
pcs, ins = pc16[sl], inm[sl]
nper = 2048
f, Ppc = welch(pcs, fs=FS, nperseg=nper)
_, Pin = welch(ins, fs=FS, nperseg=nper)
Hmag = np.sqrt(np.maximum(Pin, 1e-20) / np.maximum(Ppc, 1e-20))      # INMP / conventional

# 1/3-oct smooth + normalize to 200-500 Hz (matches Fig. 1)
def smooth(f, H, frac=3):
    out = np.empty_like(H)
    for i, fc in enumerate(f):
        if fc <= 0: out[i] = H[i]; continue
        lo, hi = fc / 2 ** (1 / (2 * frac)), fc * 2 ** (1 / (2 * frac))
        m = (f >= lo) & (f <= hi); out[i] = H[m].mean() if m.any() else H[i]
    return out
Hs = smooth(f, Hmag)
Hs /= Hs[(f >= 200) & (f <= 500)].mean()
# restrict to the trustworthy measured band; taper to unity (0 dB) outside 60-7000 Hz
band = (f >= 60) & (f <= 7000)
gains = np.ones_like(Hs)
gains[band] = Hs[band]
# FIR design over normalized 0..1 (Nyquist=8000). firwin2 needs f[0]=0, f[-1]=1.
fn = f / (FS / 2)
fn = np.clip(fn, 0, 1); fn[0], fn[-1] = 0.0, 1.0
NTAPS = 257
fir = firwin2(NTAPS, fn, gains)
np.save(os.path.expanduser("~/wuwexp/domainshift/twin_fir.npy"), fir)

# --- validate: the designed filter's response should match the target Hs ---
wv, hh = freqz(fir, worN=1024, fs=FS)
tgt = np.interp(wv, f, gains)
err = np.abs(20 * np.log10(np.abs(hh) + 1e-9) - 20 * np.log10(tgt + 1e-9))
inb = (wv >= 100) & (wv <= 6000)
print("[twin] FIR taps=%d  design error in 100-6000 Hz: mean %.2f dB, max %.2f dB"
      % (NTAPS, err[inb].mean(), err[inb].max()))
# peak emphasis the filter imposes in the speech band
sb = (wv >= 500) & (wv <= 2400)
print("[twin] filter gain in 0.5-2.4 kHz: mean %.1f dB, peak %.1f dB (target from Fig.1 ~+18 dB peak)"
      % ((20 * np.log10(np.abs(hh[sb]))).mean(), (20 * np.log10(np.abs(hh[sb]))).max()))

# apply to the conventional sweep and confirm it moves toward the INMP441 spectrum
pc_filt = lfilter(fir, 1.0, pcs)
_, Pf = welch(pc_filt, fs=FS, nperseg=nper)
def band_db(P, lo, hi): return 10 * np.log10(P[(f >= lo) & (f <= hi)].sum())
for lo, hi in [(300, 500), (500, 1000), (1000, 2400), (2400, 4000)]:
    d_before = band_db(Ppc, lo, hi); d_after = band_db(Pf, lo, hi); d_tgt = band_db(Pin, lo, hi)
    print("  %4d-%4d Hz: conv %+.1f -> filtered %+.1f  (real INMP %+.1f)  [rel]"
          % (lo, hi, d_before - band_db(Ppc, 300, 500), d_after - band_db(Pf, 300, 500), d_tgt - band_db(Pin, 300, 500)))
print("SAVED twin_fir.npy")

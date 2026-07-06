"""
exp_frontend_snr.py -- EXP A: PCEN vs log-mel on a LONG STREAM (the correct test).

Why long streams: stateful PCEN adapts its per-channel gain to the persistent
fan over time, so BEFORE the wake arrives its gain has settled on the fan and
the (quiet/distant) wake is a detectable transient above the adapted floor.
A short wake+fan clip gives PCEN no pre-wake adaptation window, so it hides the
effect (which is exactly why the clean/clip benchmark ties). log-mel has no
temporal adaptation at all.

Design: build a continuous stream = [PREFIX s of fan-only] + [quiet wake mixed
into the continuing fan at a target SNR] + [SUFFIX s of fan]. Step BOTH the
stateful-PCEN detector and the (non-adaptive) log-mel detector across the whole
stream (deployment stride + EMA), and check whether each fires in the wake
window. Sweep SNR (wake loudness vs fan) -> detection rate vs SNR per front-end.

Trains a PCEN model + a log-mel model (same Noban-V17, 3-class). FP32.

Usage:
  python exp_frontend_snr.py --epochs 40 --n-files 150
  python exp_frontend_snr.py --smoke
"""
import os, glob, json, time, argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import common as C
import train_ablation as TA
import build_cache as BC

RESULTS = os.path.expanduser("~/wuwexp/results")
PCEN_CACHE, LOGMEL_CACHE = "~/wuwexp/cache/features_pcen.npz", "~/wuwexp/cache/features_logmel.npz"
SR, NFFT, HOP, LB = C.TARGET_SR, C.N_FFT, C.HOP_LENGTH, C.LOOKBACK
NMELS, FMIN, FMAX, TW, TLEN = C.N_MELS, C.FMIN, C.FMAX, C.TARGET_WIDTH, C.TARGET_LEN
STRIDE = int(0.3 * SR)
PA, PD, PR, PT = C.PCEN_ALPHA, C.PCEN_DELTA, C.PCEN_R, C.PCEN_TIME_C
PREFIX_S, SUFFIX_S = 8.0, 2.0          # fan-only adaptation window before the wake
TH = 0.5
SNRS = [40, 20, 10, 5, 0, -5, -10, -15]  # +40 = near-clean anchor (sanity)

# ---------- training (faithful 3-class, reused) ----------
def train_3class(cache_path, epochs):
    TA.set_cache(cache_path)
    Xtr, ytr3, Xev, yev3, Xte, yte3 = TA.load_cache()
    ytr, K = TA.remap(ytr3, "3class"); yev, _ = TA.remap(yev3, "3class")
    cw = TA.class_weights(ytr, "3class")
    ytr_oh = tf.keras.utils.to_categorical(ytr, K); yev_oh = tf.keras.utils.to_categorical(yev, K)
    model = C.build_model_v17(input_shape=(50, 32, 1), num_classes=K)
    spe = len(Xtr) // 32
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    model.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
              epochs=epochs, class_weight=cw, verbose=2,
              callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True)])
    return model

def held_out_wake_files():
    paths, labels = BC.gather_paths(None)
    idx = np.arange(len(paths))
    itr, itmp = train_test_split(idx, test_size=0.20, stratify=labels, random_state=C.SEED)
    iev, ite = train_test_split(itmp, test_size=0.50, stratify=labels[itmp], random_state=C.SEED)
    return [paths[i] for i in ite if labels[i] == 0]

def _mel(sig):
    return librosa.feature.melspectrogram(y=sig, sr=SR, n_fft=NFFT, hop_length=HOP,
        n_mels=NMELS, fmin=FMIN, fmax=FMAX, power=1.0, center=False)

def _warm_zi(n=5):
    m = _mel(np.zeros(TLEN, dtype=np.float32)); zi = None
    for _ in range(n):
        _, zi = librosa.pcen(S=m * (2**31), sr=SR, hop_length=HOP, time_constant=PT,
            gain=PA, bias=PD, power=PR, eps=1e-3, zi=zi, return_zf=True)
    return zi

# ---------- streaming feature sequences over a whole stream ----------
def seq_pcen(stream):
    zi = _warm_zi(); buf = np.zeros((TW, NMELS), dtype=np.float32); lb = np.zeros(LB, dtype=np.float32); out = []
    for s in range(0, len(stream) - STRIDE + 1, STRIDE):
        chunk = stream[s:s + STRIDE].astype(np.float32)
        inp = np.concatenate([lb, chunk])
        mel = _mel(inp)
        lb = chunk[-LB:].astype(np.float32).copy() if len(chunk) >= LB else np.concatenate([lb, chunk])[-LB:]
        mel, zi = librosa.pcen(S=mel * (2**31), sr=SR, hop_length=HOP, time_constant=PT,
            gain=PA, bias=PD, power=PR, eps=1e-3, zi=zi, return_zf=True)
        nf = mel.T; n_new = nf.shape[0]
        if n_new >= TW: buf = nf[-TW:].astype(np.float32)
        elif n_new > 0:
            buf = np.roll(buf, -n_new, axis=0); buf[-n_new:] = nf
        out.append(buf.copy())
    return np.stack(out)[..., np.newaxis] if out else np.zeros((0, TW, NMELS, 1), np.float32)

def seq_logmel(stream):
    """Non-adaptive: each stride, log-mel on the last 1 s window + per-sample norm."""
    out = []
    for s in range(0, len(stream) - STRIDE + 1, STRIDE):
        end = s + STRIDE
        win = stream[max(0, end - TLEN):end].astype(np.float32)
        if len(win) < TLEN: win = np.pad(win, (TLEN - len(win), 0))
        mel = librosa.feature.melspectrogram(y=win, sr=SR, n_fft=NFFT, hop_length=HOP,
            n_mels=NMELS, fmin=FMIN, fmax=FMAX, power=2.0, center=False)
        mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0).T
        if mel.shape[0] < TW: mel = np.pad(mel, ((TW - mel.shape[0], 0), (0, 0)))
        else: mel = mel[-TW:]
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        out.append(mel.astype(np.float32))
    return np.stack(out)[..., np.newaxis] if out else np.zeros((0, TW, NMELS, 1), np.float32)

def peak_in_window(scores, s0, s1):
    """Raw max P(wake) in the wake window -- direct measure of front-end capability."""
    seg = scores[max(0, s0):s1 + 1]
    return float(np.max(seg)) if len(seg) else 0.0

def build_bg(noise_files, need_len, rng):
    buf = []
    tot = 0
    while tot < need_len:
        nz, _ = librosa.load(noise_files[rng.randint(len(noise_files))], sr=SR, mono=True)
        if nz is None or len(nz) == 0: continue
        buf.append(nz.astype(np.float32)); tot += len(nz)
    bg = np.concatenate(buf)[:need_len]
    return bg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-files", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    snrs = [10, 0, -10] if args.smoke else SNRS
    if args.smoke: args.epochs, args.n_files = 2, 4
    C.seed_everything()

    print("[expA] training PCEN model ...", flush=True); m_pcen = train_3class(PCEN_CACHE, args.epochs)
    print("[expA] training log-mel model ...", flush=True); m_log = train_3class(LOGMEL_CACHE, args.epochs)

    wake_files = held_out_wake_files()[:args.n_files]
    noise_files = sorted(glob.glob(os.path.join(C.NOISE_ROOT, "*.wav")))
    rng = np.random.RandomState(C.SEED)
    prefix, suffix = int(PREFIX_S * SR), int(SUFFIX_S * SR)
    print(f"[expA] {len(wake_files)} wake files, {len(noise_files)} noise files, stream={PREFIX_S}+wake+{SUFFIX_S}s", flush=True)

    # pre-build (clean wake, background) pairs
    pairs = []
    for i, wf in enumerate(wake_files):
        w, _ = librosa.load(wf, sr=SR, mono=True)
        if w is None or len(w) < int(0.3 * SR): continue
        w = w[:TLEN] if len(w) > TLEN else w
        bg = build_bg(noise_files, prefix + len(w) + suffix, rng)
        pairs.append((w, bg))
    print(f"[expA] built {len(pairs)} streams", flush=True)

    curve = {"pcen": {}, "logmel": {}}; meanpk = {"pcen": {}, "logmel": {}}
    for snr in snrs:
        det_p = det_l = 0; pks_p = []; pks_l = []
        for w, bg in pairs:
            stream = bg.copy()
            reg = bg[prefix:prefix + len(w)]
            ps = float(np.mean(w ** 2)) + 1e-12; pn = float(np.mean(reg ** 2)) + 1e-12
            g = np.sqrt(pn * (10.0 ** (snr / 10.0)) / ps)    # scale WAKE to target SNR vs fan
            stream[prefix:prefix + len(w)] = reg + g * w
            s0 = max(0, prefix // STRIDE - 1); s1 = (prefix + len(w)) // STRIDE + 4
            pk_p = peak_in_window(m_pcen.predict(seq_pcen(stream), batch_size=256, verbose=0)[:, 0], s0, s1)
            pk_l = peak_in_window(m_log.predict(seq_logmel(stream), batch_size=256, verbose=0)[:, 0], s0, s1)
            det_p += (pk_p > TH); det_l += (pk_l > TH); pks_p.append(pk_p); pks_l.append(pk_l)
        n = len(pairs)
        curve["pcen"][snr] = round(det_p / n, 4); curve["logmel"][snr] = round(det_l / n, 4)
        meanpk["pcen"][snr] = round(float(np.mean(pks_p)), 4); meanpk["logmel"][snr] = round(float(np.mean(pks_l)), 4)
        print(f"  SNR {snr:+4d} dB   PCEN det={det_p/n:.3f} (pk {np.mean(pks_p):.2f})   "
              f"logmel det={det_l/n:.3f} (pk {np.mean(pks_l):.2f})", flush=True)

    res = {"exp": "frontend_snr_stream", "epochs": args.epochs, "n_streams": len(pairs),
           "prefix_s": PREFIX_S, "snrs": snrs, "detection_rate": curve, "mean_peak": meanpk}
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "exp_frontend_snr.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n==== EXP A RESULT ===="); print(json.dumps(res, indent=2)); print("saved ->", out)

if __name__ == "__main__":
    main()

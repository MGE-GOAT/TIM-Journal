"""stats_review.py -- statistical rigor for the reviewer response.
(A) closed-form Clopper-Pearson / Poisson CIs for the reported false-alarm counts (no training);
(B) run-to-run variance: the 3-class FP32 model over N seeds, per-source FA counts (measures the
    'scatter' the reviewers say we invoke but never measure, e.g. the 7/0 vs 2/1 discrepancy);
(C) McNemar paired significance test on the PCEN-vs-log-mel SNR sweep (per-clip, at each SNR).
Usage: python stats_review.py --seeds 99 7 21 42 123 --epochs 25 [--smoke]
"""
import os, json, glob, argparse
import numpy as np, tensorflow as tf
import common as C, train_ablation as TA, exp_attn_interaction as EC, build_cache as BC
from sklearn.model_selection import train_test_split
try:
    from scipy.stats import beta as _beta, chi2 as _chi2
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RESULTS = os.path.expanduser("~/wuwexp/results")
PCEN = "~/wuwexp/cache/features_pcen.npz"; LOGMEL = "~/wuwexp/cache/features_logmel.npz"

def cp_ci(k, n, alpha=0.05):
    if not HAVE_SCIPY:
        p = k / n; se = (p * (1 - p) / n) ** 0.5
        return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi

def part_A():
    cells = [("charII_dscnns_speech", 18, 1552), ("charII_dscnns_noise", 2, 1552),
             ("charII_prop_speech", 7, 1552), ("charII_prop_noise", 0, 1552),
             ("charII_tcresnet_noise", 7, 1552), ("labelIII_weights_speech", 9, 1552),
             ("labelIII_3class_speech", 2, 1552), ("scarcity_binnat_speech", 42, 1552),
             ("scarcity_3class_speech", 0, 1552), ("far2frr_1event", 1, 3104)]
    out = {}
    for lbl, k, n in cells:
        lo, hi = cp_ci(k, n)
        out[lbl] = {"k": k, "n": n, "rate_pct": round(100 * k / n, 4),
                    "ci95_count": [round(lo * n, 1), round(hi * n, 1)],
                    "ci95_pct": [round(100 * lo, 4), round(100 * hi, 4)]}
    return out

def _fit3(seed, epochs):
    C.seed_everything(seed)
    Xtr, ytr3, Xev, yev3, Xte, yte3 = TA.load_cache()
    ytr_oh = tf.keras.utils.to_categorical(ytr3, 3); yev_oh = tf.keras.utils.to_categorical(yev3, 3)
    m = C.build_model_v17((50, 32, 1), 3)
    spe = max(1, len(Xtr) // 32)
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    m.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
          epochs=epochs, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(
              monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)])
    p = m.predict(Xte, batch_size=256, verbose=0)[:, 0]; fired = p >= 0.5
    return int(np.sum(fired[yte3 == 1])), int(np.sum(fired[yte3 == 2]))

def part_B(seeds, epochs):
    runs = []
    for s in seeds:
        sp, nz = _fit3(s, epochs)
        runs.append({"seed": s, "speech_FA": sp, "noise_FA": nz})
        print(f"[seedB] seed={s} speech={sp} noise={nz}", flush=True)
    sp = [r["speech_FA"] for r in runs]; nz = [r["noise_FA"] for r in runs]
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return {"runs": runs, "speech_mean": float(np.mean(sp)), "speech_sd": sd(sp), "speech_range": [min(sp), max(sp)],
            "noise_mean": float(np.mean(nz)), "noise_sd": sd(nz), "noise_range": [min(nz), max(nz)]}

def _mix(clean, noise, snr):
    ps = float(np.mean(clean ** 2)) + 1e-12; pn = float(np.mean(noise ** 2)) + 1e-12
    a = np.sqrt(pn * (10.0 ** (snr / 10.0)) / ps); return (a * clean + noise).astype(np.float32)

def _feat(audio, pcen):
    C.USE_PCEN = pcen; C.LOG_MEL = (not pcen)
    return C.format_ds_cnn(C.extract_ds_cnn_mfe(audio.astype(np.float32)))

def _held_out_wake(nf):
    paths, labels = BC.gather_paths(None); idx = np.arange(len(paths))
    _, itmp = train_test_split(idx, test_size=0.20, stratify=labels, random_state=C.SEED)
    _, ite = train_test_split(itmp, test_size=0.50, stratify=labels[itmp], random_state=C.SEED)
    return [paths[i] for i in ite if labels[i] == 0][:nf]

def _mcnemar(b, c):
    if (b + c) == 0: return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(_chi2.sf(stat, 1)) if HAVE_SCIPY else float("nan")
    return round(stat, 3), round(p, 4)

def part_C(epochs, nf, snrs):
    mp, _, _ = EC.train(PCEN, True, epochs)
    ml, _, _ = EC.train(LOGMEL, True, epochs)
    wake = _held_out_wake(nf); noise = sorted(glob.glob(os.path.join(C.NOISE_ROOT, "*.wav")))
    pairs = []
    for i, wf in enumerate(wake):
        cl = C.load_audio(wf); nz = C.load_audio(noise[i % len(noise)])
        if cl is not None and nz is not None: pairs.append((cl, nz))
    out = {}
    for snr in snrs:
        Xp = []; Xl = []
        for cl, nz in pairs:
            mx = _mix(cl, nz, snr); Xp.append(_feat(mx, True)); Xl.append(_feat(mx, False))
        pp = mp.predict(np.stack(Xp), batch_size=256, verbose=0)[:, 0] >= 0.5
        pl = ml.predict(np.stack(Xl), batch_size=256, verbose=0)[:, 0] >= 0.5
        b = int(np.sum(pp & ~pl)); c = int(np.sum(~pp & pl))
        chi2, p = _mcnemar(b, c)
        out[str(snr)] = {"n": len(pairs), "pcen_rate": round(float(np.mean(pp)), 3),
                         "logmel_rate": round(float(np.mean(pl)), 3), "pcen_only": b,
                         "logmel_only": c, "mcnemar_chi2": chi2, "p": p}
        print(f"[mcnemar] snr={snr} pcen={np.mean(pp):.3f} logmel={np.mean(pl):.3f} b={b} c={c} chi2={chi2} p={p}", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[99, 7, 21, 42, 123])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.seeds, a.epochs = [99, 7], 2
    C.seed_everything(); TA.set_cache(PCEN)
    snrs = [10, 5, 0, -5] if not a.smoke else [0]
    res = {"scipy": HAVE_SCIPY, "ci": part_A(),
           "seed_variance": part_B(a.seeds, a.epochs),
           "mcnemar_snr": part_C(a.epochs, 400 if not a.smoke else 20, snrs)}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "stats_review.json"), "w"), indent=2)
    print("\n==== STATS RESULT ====\n" + json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

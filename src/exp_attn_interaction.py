"""
exp_attn_interaction.py -- EXP C: front-end x attention INTERACTION.

Tests the applicant's claim that coordinate/T-F attention (+SE) exploits PCEN's
structure. Reviewer risk: PCEN already does per-channel AGC, so the attention may
be REDUNDANT with it. Only a measured INTERACTION settles it.

2x2: {PCEN, log-mel} x {attention ON (coord-attn + SE), attention OFF}.
Train all four (same Noban-V17 backbone; attention blocks -> identity when OFF),
then run Exp A's streaming SNR/far-field sweep on each. Report:
  attention gain for PCEN  = det[PCEN,ON]  - det[PCEN,OFF]
  attention gain for logmel= det[log,ON]   - det[log,OFF]
Synergy (supports the claim) if PCEN's gain > log-mel's gain, esp. at low SNR.
If gains are equal / attention hurts PCEN -> redundant, drop the claim.

Reuses exp_frontend_snr.py for the streaming sweep machinery.

Usage:
  python exp_attn_interaction.py --epochs 40 --n-files 120
  python exp_attn_interaction.py --smoke
"""
import os, json, argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import common as C
import train_ablation as TA
import exp_frontend_snr as EA

RESULTS = os.path.expanduser("~/wuwexp/results")
PCEN_CACHE, LOGMEL_CACHE = "~/wuwexp/cache/features_pcen.npz", "~/wuwexp/cache/features_logmel.npz"

def build_v17(num_classes, attention):
    """Noban-V17 with attention blocks made identity when attention=False."""
    if attention:
        return C.build_model_v17(input_shape=(50, 32, 1), num_classes=num_classes)
    orig_ca, orig_se = C.coord_attention, C.se_block
    C.coord_attention = lambda x, *a, **k: x
    C.se_block = lambda x, *a, **k: x
    try:
        m = C.build_model_v17(input_shape=(50, 32, 1), num_classes=num_classes)
    finally:
        C.coord_attention, C.se_block = orig_ca, orig_se
    return m

def train(cache_path, attention, epochs):
    TA.set_cache(cache_path)
    Xtr, ytr3, Xev, yev3, Xte, yte3 = TA.load_cache()
    ytr, K = TA.remap(ytr3, "3class"); yev, _ = TA.remap(yev3, "3class")
    cw = TA.class_weights(ytr, "3class")
    ytr_oh = tf.keras.utils.to_categorical(ytr, K); yev_oh = tf.keras.utils.to_categorical(yev, K)
    model = build_v17(K, attention)
    spe = len(Xtr) // 32
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    model.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
              epochs=epochs, class_weight=cw, verbose=2,
              callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True)])
    return model, K, (Xte, yte3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-files", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    snrs = [10, 0, -10] if args.smoke else [20, 10, 5, 0, -5, -10, -15]
    if args.smoke: args.epochs, args.n_files = 2, 4
    C.seed_everything()

    configs = [("pcen", PCEN_CACHE, True), ("pcen", PCEN_CACHE, False),
               ("logmel", LOGMEL_CACHE, True), ("logmel", LOGMEL_CACHE, False)]
    models = {}; clean = {}
    for front, cache, att in configs:
        key = f"{front}_{'att' if att else 'noatt'}"
        print(f"[expC] training {key} ...", flush=True)
        m, K, (Xte, yte3) = train(cache, att, args.epochs)
        models[key] = (m, front)
        clean[key] = TA.evaluate(m, Xte, yte3, K)
        print(f"  [{key}] clean AUC={clean[key]['wake_auc']:.5f} FAR@2%={clean[key]['far_at_frr2pct']:.4f} "
              f"params={m.count_params()}", flush=True)

    # streaming SNR sweep (reuse Exp A machinery) -- build streams once, score all 4
    wake_files = EA.held_out_wake_files()[:args.n_files]
    noise_files = sorted(__import__("glob").glob(os.path.join(C.NOISE_ROOT, "*.wav")))
    rng = np.random.RandomState(C.SEED)
    prefix, suffix = int(EA.PREFIX_S * EA.SR), int(EA.SUFFIX_S * EA.SR)
    pairs = []
    for wf in wake_files:
        w, _ = librosa.load(wf, sr=EA.SR, mono=True)
        if w is None or len(w) < int(0.3 * EA.SR): continue
        w = w[:EA.TLEN] if len(w) > EA.TLEN else w
        pairs.append((w, EA.build_bg(noise_files, prefix + len(w) + suffix, rng)))
    print(f"[expC] {len(pairs)} streams", flush=True)

    det = {k: {} for k in models}
    for snr in snrs:
        acc = {k: 0 for k in models}
        for w, bg in pairs:
            stream = bg.copy(); reg = bg[prefix:prefix + len(w)]
            ps = float(np.mean(w ** 2)) + 1e-12; pn = float(np.mean(reg ** 2)) + 1e-12
            g = np.sqrt(pn * (10.0 ** (snr / 10.0)) / ps)
            stream[prefix:prefix + len(w)] = reg + g * w
            s0 = max(0, prefix // EA.STRIDE - 1); s1 = (prefix + len(w)) // EA.STRIDE + 4
            Xp = EA.seq_pcen(stream); Xl = EA.seq_logmel(stream)
            for k, (m, front) in models.items():
                X = Xp if front == "pcen" else Xl
                acc[k] += (EA.peak_in_window(m.predict(X, batch_size=256, verbose=0)[:, 0], s0, s1) > EA.TH)
        n = len(pairs)
        for k in models: det[k][snr] = round(acc[k] / n, 4)
        print(f"  SNR {snr:+4d}: " + "  ".join(f"{k}={det[k][snr]:.2f}" for k in models), flush=True)

    interaction = {}
    for snr in snrs:
        gp = det["pcen_att"][snr] - det["pcen_noatt"][snr]
        gl = det["logmel_att"][snr] - det["logmel_noatt"][snr]
        interaction[snr] = {"pcen_att_gain": round(gp, 4), "logmel_att_gain": round(gl, 4),
                            "interaction(pcen-logmel)": round(gp - gl, 4)}
    res = {"exp": "attn_interaction", "epochs": args.epochs, "n_streams": len(pairs),
           "clean": clean, "detection_rate": det, "interaction": interaction}
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "exp_attn_interaction.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n==== EXP C RESULT ===="); print(json.dumps(res, indent=2)); print("saved ->", out)

if __name__ == "__main__":
    main()

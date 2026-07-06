"""
exp_noise_robust.py -- CORRECT in-distribution front-end x attention robustness.

Replaces the broken streaming Exp A/C. Adds noise DIRECTLY to the standard 2.4s
clips (the exact distribution the model was trained on: C.load_audio -> pad/trim
to 2.4s -> C.extract_ds_cnn_mfe), so the model behaves normally (a clean wake
detects ~100%, not the broken 13%). Measures detection-rate vs SNR for the 2x2
{PCEN, log-mel} x {attention on, off}. Answers: does PCEN beat log-mel under
noise, and does attention help PCEN more (interaction)?
Honest scope: clip-level NOISE robustness, not long-fan-adaptation.

Usage: python exp_noise_robust.py --epochs 50 --n-files 400 [--smoke]
"""
import os, glob, json, argparse
import numpy as np, librosa, tensorflow as tf
from sklearn.model_selection import train_test_split
import common as C, train_ablation as TA, build_cache as BC
import exp_attn_interaction as EC          # reuse train(cache,att,epochs) + build_v17

RESULTS = os.path.expanduser("~/wuwexp/results")
PCEN, LOGMEL = "~/wuwexp/cache/features_pcen.npz", "~/wuwexp/cache/features_logmel.npz"
SNRS = [30, 20, 10, 5, 0, -5, -10, -15]

def held_out_wake_files():
    paths, labels = BC.gather_paths(None)
    idx = np.arange(len(paths))
    itr, itmp = train_test_split(idx, test_size=0.20, stratify=labels, random_state=C.SEED)
    iev, ite = train_test_split(itmp, test_size=0.50, stratify=labels[itmp], random_state=C.SEED)
    return [paths[i] for i in ite if labels[i] == 0]

def feat(audio, pcen):
    C.USE_PCEN = pcen; C.LOG_MEL = (not pcen)
    return C.format_ds_cnn(C.extract_ds_cnn_mfe(audio.astype(np.float32)))

def mix(clean, noise, snr):
    # Far-field simulation: keep the fan (noise) at its natural level and ATTENUATE
    # the WAKE to the target SNR (lower SNR = quieter, more-distant voice). This is
    # more faithful to far-field than amplifying the noise, and it exercises PCEN's
    # adaptive gain on a quiet signal -- the exact mechanism claimed in deployment.
    ps = float(np.mean(clean ** 2)) + 1e-12; pn = float(np.mean(noise ** 2)) + 1e-12
    a = np.sqrt(pn * (10.0 ** (snr / 10.0)) / ps)       # scale WAKE down to target SNR
    return (a * clean + noise).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-files", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    snrs = [20, 0, -15] if args.smoke else SNRS
    if args.smoke: args.epochs, args.n_files = 2, 20
    C.seed_everything()

    configs = [("pcen", PCEN, True), ("pcen", PCEN, False),
               ("logmel", LOGMEL, True), ("logmel", LOGMEL, False)]
    models = {}; clean = {}
    for front, cache, att in configs:
        key = f"{front}_{'att' if att else 'noatt'}"
        print(f"[expN] training {key} ...", flush=True)
        m, K, (Xte, yte3) = EC.train(cache, att, args.epochs)
        models[key] = (m, front); clean[key] = TA.evaluate(m, Xte, yte3, K)
        print(f"  [{key}] clean AUC={clean[key]['wake_auc']:.5f}", flush=True)

    wake_files = held_out_wake_files()[:args.n_files]
    noise_files = sorted(glob.glob(os.path.join(C.NOISE_ROOT, "*.wav")))
    pairs = []
    for i, wf in enumerate(wake_files):
        cl = C.load_audio(wf); nz = C.load_audio(noise_files[i % len(noise_files)])
        if cl is not None and nz is not None: pairs.append((cl, nz))
    print(f"[expN] {len(pairs)} clean+noise pairs", flush=True)

    det = {k: {} for k in models}
    for snr in snrs:
        Xp = []; Xl = []
        for cl, nz in pairs:
            mx = mix(cl, nz, snr); Xp.append(feat(mx, True)); Xl.append(feat(mx, False))
        Xp = np.stack(Xp); Xl = np.stack(Xl)
        for k, (m, front) in models.items():
            X = Xp if front == "pcen" else Xl
            det[k][snr] = round(float(np.mean(m.predict(X, batch_size=256, verbose=0)[:, 0] > 0.5)), 4)
        print(f"  SNR {snr:+4d}: " + "  ".join(f"{k}={det[k][snr]:.3f}" for k in models), flush=True)

    inter = {snr: {"pcen_att_gain": round(det["pcen_att"][snr] - det["pcen_noatt"][snr], 4),
                   "logmel_att_gain": round(det["logmel_att"][snr] - det["logmel_noatt"][snr], 4),
                   "pcen_vs_logmel(att)": round(det["pcen_att"][snr] - det["logmel_att"][snr], 4)} for snr in snrs}
    res = {"exp": "noise_robust_clip", "epochs": args.epochs, "n": len(pairs),
           "clean": clean, "detection_rate": det, "interaction": inter}
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "exp_noise_robust.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n==== EXP N RESULT ===="); print(json.dumps(res, indent=2)); print("saved ->", out)

if __name__ == "__main__":
    main()

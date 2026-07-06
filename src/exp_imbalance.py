"""
exp_imbalance.py -- keyword-class SCARCITY ablation (the "custom keyword, few
wake utterances" scenario). Induces wake-class scarcity in TRAIN only (val/test
stay full), then compares label/rebalancing schemes on BOTH detection (wake
recall, AUC) and false alarms. Answers honestly whether the 3-class formulation
holds up under scarcity WITHOUT oversampling or class weights, or whether only
oversampling recovers detection.

Schemes: 3class-natural, binary-natural, binary-classweights, binary-oversample.
Usage: python exp_imbalance.py --n-wake 500 --epochs 30 [--smoke]
"""
import os, json, argparse
import numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score
import common as C, train_ablation as TA

RESULTS = os.path.expanduser("~/wuwexp/results")

def subsample_wake(X, y, n_wake, seed=C.SEED):
    rng = np.random.RandomState(seed)
    wake = np.where(y == 0)[0]; notw = np.where(y != 0)[0]
    keep = rng.choice(wake, size=min(n_wake, len(wake)), replace=False)
    idx = np.concatenate([keep, notw]); rng.shuffle(idx)
    return X[idx], y[idx]

def oversample_wake(X, y, seed=C.SEED + 1):
    rng = np.random.RandomState(seed)
    wake = np.where(y == 0)[0]; notw = np.where(y != 0)[0]
    need = len(notw) - len(wake)
    reps = rng.choice(wake, size=max(0, need), replace=True)
    idx = np.concatenate([wake, reps, notw]); rng.shuffle(idx)
    return X[idx], y[idx]

def recall_at(model, X, y3, thr=0.5):
    p = model.predict(X, batch_size=256, verbose=0)[:, 0]
    is_wake = (y3 == 0)
    return float(np.mean(p[is_wake] >= thr))

def train_one(Xtr, ytr3, Xev, yev3, K, cw, epochs):
    ytr = ytr3 if K == 3 else (ytr3 != 0).astype(np.int64)
    yev = yev3 if K == 3 else (yev3 != 0).astype(np.int64)
    ytr_oh = tf.keras.utils.to_categorical(ytr, K)
    yev_oh = tf.keras.utils.to_categorical(yev, K)
    m = C.build_model_v17(input_shape=(50, 32, 1), num_classes=K)
    spe = max(1, len(Xtr) // 32)
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    m.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
          epochs=epochs, class_weight=cw, verbose=2,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                     patience=10, restore_best_weights=True)])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wake", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_wake = 2, 200
    C.seed_everything()
    TA.set_cache("~/wuwexp/cache/features_pcen.npz")
    Xtr, ytr3, Xev, yev3, Xte, yte3 = TA.load_cache()
    n_full = int((ytr3 == 0).sum())
    Xs, ys = subsample_wake(Xtr, ytr3, args.n_wake)
    Xo, yo = oversample_wake(Xs, ys)
    print(f"[imb] wake {n_full} -> {args.n_wake}; natural train={Xs.shape}, oversampled={Xo.shape}", flush=True)

    def bal_cw(y):
        yb = (y != 0).astype(np.int64)
        cls, cnt = np.unique(yb, return_counts=True)
        return {int(c): float(len(yb) / (len(cls) * n)) for c, n in zip(cls, cnt)}

    configs = [("3class_natural", 3, None, Xs, ys),
               ("binary_natural", 2, None, Xs, ys),
               ("binary_weights", 2, bal_cw(ys), Xs, ys),
               ("binary_oversample", 2, None, Xo, yo)]
    out = {}
    for name, K, cw, Xc, yc in configs:
        print(f"[imb] === {name} K={K} cw={cw} n={len(Xc)} ===", flush=True)
        m = train_one(Xc, yc, Xev, yev3, K, cw, args.epochs)
        r = TA.evaluate(m, Xte, yte3, K)
        r["wake_recall@0.5"] = recall_at(m, Xte, yte3)
        out[name] = r
        print(f"  [{name}] AUC={r['wake_auc']:.5f} FAR@2%={r['far_at_frr2pct']:.4f} "
              f"recall@0.5={r['wake_recall@0.5']:.4f} FARnoise={r['far_noise@0.5']:.4f} "
              f"FARspeech={r['far_other@0.5']:.4f}", flush=True)

    res = {"exp": "keyword_scarcity", "n_wake": args.n_wake, "n_wake_full": n_full,
           "epochs": args.epochs, "schemes": out}
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "exp_imbalance.json")
    json.dump(res, open(p, "w"), indent=2)
    print("\n==== EXP IMBALANCE RESULT ===="); print(json.dumps(res, indent=2)); print("saved ->", p)

if __name__ == "__main__":
    main()

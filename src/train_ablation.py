"""
train_ablation.py — the 3-class vs binary vs FRR-target ablation.

Trains the SAME Noban-V17 arch on the SAME cached features under three label
schemes, then scores all three on a COMMON wake-vs-not-wake metric so they are
directly comparable:

  scheme=3class : 3-class softmax (wake/other/noise), balanced CE + label-smooth
                  -> the thesis setup (FRR used only for monitoring).
  scheme=binary : 2-class (wake vs not-wake), other+noise collapsed. Tests
                  "does dropping the other/noise distinction hurt?"
  scheme=frr    : 2-class wake-vs-rest trained to DIRECTLY chase the asymmetric
                  deployment cost (FAR penalised FT_PENALTY x via class weight).
                  Tests "does optimising the FRR/FAR criterion directly hurt?"

Metrics (all schemes, on the test split):
  * wake-vs-not-wake ROC-AUC   (threshold-free, primary)
  * FAR @ FRR<=2%              (fixed operating point)
  * FAR broken out on NOISE negatives vs OTHER negatives (the "noisy env" angle)
  * 3class only: other-vs-noise separation accuracy (the distinction binary loses)

Usage:
  python train_ablation.py --scheme 3class --epochs 60 --tag run1
  python train_ablation.py --scheme binary --qat
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import common as C

CACHE = os.path.expanduser("~/wuwexp/cache/features_pcen.npz")
RESULTS_DIR = os.path.expanduser("~/wuwexp/results")

def set_cache(path):
    global CACHE
    CACHE = os.path.expanduser(path)
FT_PENALTY = 5.0        # FAR penalty in the deployment cost FRR + FT_PENALTY*FAR
SPECAUG = True
TIME_MASK_MAX, FREQ_MASK_MAX = 10, 4

# ---------------- data ----------------
def load_cache():
    d = np.load(CACHE)
    return d["Xtr"], d["ytr"], d["Xev"], d["yev"], d["Xte"], d["yte"]

def remap(y, scheme):
    """Return (labels, num_classes). Wake=0 in the 3-class cache."""
    if scheme == "3class":
        return y.astype(np.int64), 3
    # binary / frr: wake(0) -> 0 ; other(1)+noise(2) -> 1 (not-wake)
    yb = (y != 0).astype(np.int64)
    return yb, 2

def class_weights(y, scheme):
    import numpy as np
    classes, counts = np.unique(y, return_counts=True)
    if scheme == "frr":
        # directly encode the asymmetric deployment cost: penalise firing on
        # not-wake (FAR) FT_PENALTY x -> pushes the wake decision boundary,
        # ignoring internal other/noise structure. This is the "optimise the
        # FRR/FAR criterion directly" arm.
        return {0: 1.0, 1: float(FT_PENALTY)}
    # 3class / binary: balanced
    n = len(y); k = len(classes)
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}

# ---------------- SpecAugment (port of cell 3) ----------------
def tf_spec_augment(x, n_time=2, n_freq=2):
    T = tf.shape(x)[1]; F = tf.shape(x)[2]
    fill = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    def apply_mask(x_in, axis_len, max_w, axis):
        w  = tf.random.uniform([], 0, max_w + 1, dtype=tf.int32)
        w0 = tf.random.uniform([], 0, axis_len - w + 1, dtype=tf.int32)
        m  = tf.concat([tf.ones([w0]), tf.zeros([w]), tf.ones([axis_len - w0 - w])], 0)
        shape = [1, 1, 1, 1]; shape[axis] = -1
        m = tf.reshape(m, shape)
        return x_in * m + fill * (1.0 - m)
    for _ in range(n_time): x = apply_mask(x, T, TIME_MASK_MAX, 1)
    for _ in range(n_freq): x = apply_mask(x, F, FREQ_MASK_MAX, 2)
    return x

def make_ds(X, yoh, bs, augment):
    ds = tf.data.Dataset.from_tensor_slices((X, yoh))
    if augment:
        ds = ds.shuffle(4096, seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(bs)
    if augment and SPECAUG:
        ds = ds.map(lambda x, y: (tf_spec_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# ---------------- eval (common metric) ----------------
def wake_scores(model, X, num_classes):
    p = model.predict(X, batch_size=256, verbose=0)
    return p[:, 0]                       # P(wake) in both 2- and 3-class (wake=0)

def far_at_frr(wake_p, is_wake, target_frr=0.02):
    """Lowest FAR achievable while keeping FRR<=target, sweeping threshold."""
    ths = np.unique(wake_p)
    best = 1.0
    for t in ths:
        fired = wake_p >= t
        frr = np.mean(~fired[is_wake]) if is_wake.any() else 0.0
        far = np.mean(fired[~is_wake]) if (~is_wake).any() else 0.0
        if frr <= target_frr:
            best = min(best, far)
    return best

def evaluate(model, Xte, yte3, num_classes):
    is_wake = (yte3 == 0)
    wp = wake_scores(model, Xte, num_classes)
    auc = roc_auc_score(is_wake.astype(int), wp)
    far2 = far_at_frr(wp, is_wake, 0.02)
    # FAR split by negative type at a common threshold (0.5 on wake score)
    fired = wp >= 0.5
    far_noise = float(np.mean(fired[yte3 == 2])) if (yte3 == 2).any() else None
    far_other = float(np.mean(fired[yte3 == 1])) if (yte3 == 1).any() else None
    res = {"wake_auc": float(auc), "far_at_frr2pct": float(far2),
           "far_noise@0.5": far_noise, "far_other@0.5": far_other}
    # 3-class only: does it still separate other vs noise?
    if num_classes == 3:
        p = model.predict(Xte, batch_size=256, verbose=0)
        pred = p.argmax(1)
        mask = (yte3 == 1) | (yte3 == 2)
        res["other_vs_noise_acc"] = float(np.mean(pred[mask] == yte3[mask]))
    return res

# ---------------- train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["3class", "binary", "frr"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--export-tflite", action="store_true", help="save int8 tflite + keras model")
    ap.add_argument("--cache", default=None, help="feature cache path (default: PCEN cache)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cache:
        set_cache(args.cache)
    C.seed_everything()

    Xtr, ytr3, Xev, yev3, Xte, yte3 = load_cache()
    ytr, K = remap(ytr3, args.scheme)
    yev, _ = remap(yev3, args.scheme)
    cw = class_weights(ytr, args.scheme)
    print(f"[{args.scheme}] K={K}  train={Xtr.shape}  class_weight={cw}", flush=True)

    ytr_oh = tf.keras.utils.to_categorical(ytr, K)
    yev_oh = tf.keras.utils.to_categorical(yev, K)

    model = C.build_model_v17(input_shape=(50, 32, 1), num_classes=K)
    spe = len(Xtr) // 32
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
    tr = make_ds(Xtr, ytr_oh, 32, True)
    ev = make_ds(Xev, yev_oh, 32, False)
    t0 = time.time()
    model.fit(tr, validation_data=ev, epochs=args.epochs, class_weight=cw, verbose=2,
              callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=12,
                                       restore_best_weights=True)])
    fp32_time = time.time() - t0

    phase = "fp32"
    if args.qat:
        import tensorflow_model_optimization as tfmot
        def annotate(l):
            if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)):
                return tfmot.quantization.keras.quantize_annotate_layer(l)
            return l
        am = tf.keras.models.clone_model(model, clone_function=annotate)
        with tfmot.quantization.keras.quantize_scope({}):
            model = tfmot.quantization.keras.quantize_apply(am)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                      metrics=["accuracy"])
        model.fit(make_ds(Xtr, ytr_oh, 16, True), validation_data=make_ds(Xev, yev_oh, 16, False),
                  epochs=30, class_weight=cw, verbose=2,
                  callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
                             ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=7, min_lr=1e-7, mode="max")])
        phase = "qat"

    res = evaluate(model, Xte, yte3, K)
    res.update({"scheme": args.scheme, "phase": phase, "epochs": args.epochs,
                "fp32_train_s": round(fp32_time, 1), "num_classes": K,
                "params": int(model.count_params())})

    if args.export_tflite:
        import tflite_utils
        mdir = os.path.expanduser("~/wuwexp/models")
        os.makedirs(mdir, exist_ok=True)
        stub = f"{args.scheme}_{phase}{('_'+args.tag) if args.tag else ''}"
        tfl = os.path.join(mdir, f"{stub}_int8.tflite")
        nbytes = tflite_utils.to_int8_tflite(model, Xtr, tfl)
        res["tflite"] = tfl
        res["tflite_kb"] = round(nbytes / 1024, 1)
        print(f"[export] int8 tflite -> {tfl} ({nbytes/1024:.1f} KB)")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"ablation_{args.scheme}_{phase}{('_'+args.tag) if args.tag else ''}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n==== RESULT ====")
    print(json.dumps(res, indent=2))
    print("saved ->", out)

if __name__ == "__main__":
    main()

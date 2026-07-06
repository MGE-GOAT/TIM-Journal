"""
train_baselines.py — train standard KWS baselines (3-class) on the SAME cached
features + SAME recipe as Noban-V17, for the "compared to what" table.

Reports for each: #params, test accuracy, wake-vs-not-wake AUC, FAR@FRR2%,
other-vs-noise separation. Directly comparable to the Noban-V17 numbers.

Usage:  python train_baselines.py --models dscnn_s dscnn_m cnn_trad tc_resnet8 --epochs 60
"""
import os, json, time, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import common as C
from train_ablation import load_cache, make_ds, evaluate, RESULTS_DIR
from models_baseline import REGISTRY

def train_one(name, Xtr, ytr, Xev, yev, Xte, yte3, epochs):
    C.seed_everything()
    K = 3
    ytr_oh = tf.keras.utils.to_categorical(ytr, K)
    yev_oh = tf.keras.utils.to_categorical(yev, K)
    model = REGISTRY[name](K)
    params = int(model.count_params())
    spe = len(Xtr) // 32
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
    t0 = time.time()
    model.fit(make_ds(Xtr, ytr_oh, 32, True), validation_data=make_ds(Xev, yev_oh, 32, False),
              epochs=epochs, verbose=2,
              callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True)])
    dt = time.time() - t0
    # overall 3-class test acc
    pred = model.predict(Xte, batch_size=256, verbose=0).argmax(1)
    acc = float(np.mean(pred == yte3))
    res = evaluate(model, Xte, yte3, K)
    res.update({"model": name, "params": params, "test_acc": acc,
                "train_s": round(dt, 1), "epochs": epochs})
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REGISTRY.keys()))
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    Xtr, ytr, Xev, yev, Xte, yte3 = load_cache()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in args.models:
        print(f"\n######## BASELINE {name} ########", flush=True)
        res = train_one(name, Xtr, ytr, Xev, yev, Xte, yte3, args.epochs)
        out = os.path.join(RESULTS_DIR, f"baseline_{name}.json")
        json.dump(res, open(out, "w"), indent=2)
        print(json.dumps(res, indent=2)); print("saved ->", out, flush=True)

if __name__ == "__main__":
    main()

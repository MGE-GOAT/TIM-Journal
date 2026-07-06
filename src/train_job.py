"""train_job.py -- train ONE (train-cache, seed) 3-class model under a capped GPU
memory footprint so 2 jobs can share one GPU, evaluate on the given test caches,
and write a result JSON. Used by dispatch_ladder.py across PC + laptop GPUs."""
import os, json, argparse
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--train-cache", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--gpu-mem-mb", type=int, default=2600)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--out", required=True)
ap.add_argument("--test-caches", default="conv=~/wuwexp/cache/features_pcen.npz,twin=~/wuwexp/cache/features_pcen_twin.npz")
a = ap.parse_args()

# cap GPU memory BEFORE any allocation so two jobs coexist on one card
gs = tf.config.list_physical_devices("GPU")
if gs:
    try:
        tf.config.set_logical_device_configuration(
            gs[0], [tf.config.LogicalDeviceConfiguration(memory_limit=a.gpu_mem_mb)])
    except Exception as e:
        print("gpu-cap warn:", e, flush=True)

import numpy as np
import common as C, train_ablation as TA

def load(p): return np.load(os.path.expanduser(p))
tests = {}
for kv in a.test_caches.split(","):
    k, v = kv.split("="); d = load(v); tests[k] = (d["Xte"], d["yte"])

C.seed_everything(a.seed)
d = load(a.train_cache)
Xtr, ytr, Xev, yev = d["Xtr"], d["ytr"], d["Xev"], d["yev"]
ytr_oh = tf.keras.utils.to_categorical(ytr, 3); yev_oh = tf.keras.utils.to_categorical(yev, 3)
m = C.build_model_v17((50, 32, 1), 3); spe = max(1, len(Xtr) // 32)
lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
m.compile(optimizer=tf.keras.optimizers.Adam(lr, jit_compile=False),
          loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
m.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
      epochs=a.epochs, verbose=0,
      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)])
res = {tk: TA.evaluate(m, Xte, yte, 3) for tk, (Xte, yte) in tests.items()}
res.update({"label": a.label, "train_cache": a.train_cache, "seed": a.seed})
os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True)
json.dump(res, open(os.path.expanduser(a.out), "w"), indent=2)
print("JOB_DONE", a.label, "seed", a.seed, "->", a.out, flush=True)

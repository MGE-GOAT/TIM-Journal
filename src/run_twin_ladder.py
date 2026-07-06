"""run_twin_ladder.py -- cross-domain train x test matrix for the digital-twin idea.
Train the 3-class V17 on the conventional corpus (A0) and on the twin-filtered
INMP441-domain corpus (A2), and evaluate EACH on both the conventional and the
twin test splits (identical clips, different capture domain). The A0-on-twin cell
is the untreated microphone-domain-shift cost; A2-on-twin is the recovery."""
import os, json, numpy as np, tensorflow as tf
import common as C, train_ablation as TA

def load(p):
    d = np.load(os.path.expanduser(p))
    return d["Xtr"], d["ytr"], d["Xev"], d["yev"], d["Xte"], d["yte"]

CACHES = {"conv": "~/wuwexp/cache/features_pcen.npz", "twin": "~/wuwexp/cache/features_pcen_twin.npz"}
TESTS = {k: load(v)[4:6] for k, v in CACHES.items()}   # (Xte, yte) per domain

def train_one(train_key, seed):
    C.seed_everything(seed)
    Xtr, ytr, Xev, yev, _, _ = load(CACHES[train_key])
    ytr_oh = tf.keras.utils.to_categorical(ytr, 3); yev_oh = tf.keras.utils.to_categorical(yev, 3)
    m = C.build_model_v17((50, 32, 1), 3); spe = max(1, len(Xtr) // 32)
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr, jit_compile=False),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    m.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
          epochs=40, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)])
    return {tk: TA.evaluate(m, Xte, yte, 3) for tk, (Xte, yte) in TESTS.items()}

SEEDS = [99, 7, 21]
ARMS = {"A0_conv": "conv", "A2_twin": "twin"}
out = {}
for arm, tk in ARMS.items():
    out[arm] = []
    for s in SEEDS:
        r = train_one(tk, s); out[arm].append({"seed": s, **r})
        print("%s seed %d | conv: AUC %.5f FAR@2 %.4f speechFA %d | twin: AUC %.5f FAR@2 %.4f speechFA %d"
              % (arm, s, r["conv"]["wake_auc"], r["conv"]["far_at_frr2pct"], round(r["conv"]["far_other@0.5"] * 1552),
                 r["twin"]["wake_auc"], r["twin"]["far_at_frr2pct"], round(r["twin"]["far_other@0.5"] * 1552)), flush=True)
os.makedirs(os.path.expanduser("~/wuwexp/results"), exist_ok=True)
json.dump(out, open(os.path.expanduser("~/wuwexp/results/twin_ladder.json"), "w"), indent=2)

print("\n==== TRAIN x TEST MATRIX (mean over seeds) ====")
print("  arm\\test        conv-AUC  conv-FAR@2   twin-AUC  twin-FAR@2")
for arm in ARMS:
    row = out[arm]
    def mean(tk, key): return float(np.mean([x[tk][key] for x in row]))
    print("  %-12s   %.5f   %6.3f%%     %.5f   %6.3f%%"
          % (arm, mean("conv", "wake_auc"), 100 * mean("conv", "far_at_frr2pct"),
             mean("twin", "wake_auc"), 100 * mean("twin", "far_at_frr2pct")))
print("TWIN_LADDER_DONE")

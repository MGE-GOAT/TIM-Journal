"""
exp_labelscheme_silence.py -- EXP B: 3-class vs binary variants, on silence + detection.

Deployment observation to quantify: with a BINARY (wake vs not-wake) head the
model hallucinates on silence (needs a VAD); the 3-class (wake/speech/noise) head
does not, because the explicit noise class absorbs silence. Binary also inherits a
severe wake-vs-everything imbalance that must be fought with oversampling (needs
lots of wake data) or class weights (which degrade) -- neither resource-friendly
for a CUSTOM wake word. 3-class sidesteps all of this. On clean data the schemes
tie; the difference shows on SILENCE.

Four models on the SAME PCEN cache (FP32):
  3class            : 3-class, balanced class weights (proposed)
  binary_weights    : 2-class, balanced class weights
  binary_oversample : 2-class, wake oversampled to balance, NO weights
  binary_natural    : 2-class, natural imbalance, NO weights (naive baseline)

Metrics per model: wake-AUC, FAR@2%FRR, far_other/far_noise (test split, via TA),
and SILENCE activation rate (fraction of silence/quiet clips that fire P(wake)>0.5).

Usage:
  python exp_labelscheme_silence.py --epochs 40 --n-silence 600
  python exp_labelscheme_silence.py --smoke
"""
import os, json, glob, argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import common as C
import train_ablation as TA

RESULTS = os.path.expanduser("~/wuwexp/results")
PCEN_CACHE = "~/wuwexp/cache/features_pcen.npz"

def oversample_wake(X, y):
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    reps = int(np.ceil(len(idx1) / max(1, len(idx0))))
    idx0_os = np.tile(idx0, reps)[:len(idx1)]
    keep = np.concatenate([idx0_os, idx1])
    rng = np.random.RandomState(C.SEED); rng.shuffle(keep)
    return X[keep], y[keep]

def train(scheme_key, epochs):
    TA.set_cache(PCEN_CACHE)
    Xtr, ytr3, Xev, yev3, Xte, yte3 = TA.load_cache()
    if scheme_key == "3class":
        ytr, yev, K = ytr3.astype(np.int64), yev3.astype(np.int64), 3
        cw = TA.class_weights(ytr, "3class")
    else:
        ytr = (ytr3 != 0).astype(np.int64); yev = (yev3 != 0).astype(np.int64); K = 2
        if scheme_key == "binary_weights":
            cw = TA.class_weights(ytr, "binary")
        elif scheme_key == "binary_oversample":
            Xtr, ytr = oversample_wake(Xtr, ytr); cw = None
        elif scheme_key == "binary_natural":
            cw = None
        else:
            raise ValueError(scheme_key)
    print(f"[{scheme_key}] K={K} train={Xtr.shape} cw={cw}", flush=True)
    ytr_oh = tf.keras.utils.to_categorical(ytr, K); yev_oh = tf.keras.utils.to_categorical(yev, K)
    model = C.build_model_v17(input_shape=(50, 32, 1), num_classes=K)
    spe = len(Xtr) // 32
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    model.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
              epochs=epochs, class_weight=cw, verbose=2,
              callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True)])
    return model, K, (Xte, yte3)

def feat_from_audio(a):
    C.USE_PCEN = True; C.LOG_MEL = False
    return C.format_ds_cnn(C.extract_ds_cnn_mfe(a.astype(np.float32)))

def make_silence(n, rng):
    """Silence/quiet battery: digital zeros, low white noise, quiet-scaled real ambient."""
    noise_files = sorted(glob.glob(os.path.join(C.NOISE_ROOT, "*.wav")))
    clips = []
    for _ in range(n):
        t = rng.randint(4)
        if t == 0:
            a = np.zeros(C.STREAM_LEN, dtype=np.float32)
        elif t in (1, 2):
            amp = 10 ** (rng.uniform(-3.5, -2.0))            # ~ -70..-40 dBFS white
            a = (rng.randn(C.STREAM_LEN) * amp).astype(np.float32)
        else:
            nz = C.load_audio(noise_files[rng.randint(len(noise_files))])
            if nz is None: a = np.zeros(C.STREAM_LEN, dtype=np.float32)
            else:
                rms = np.sqrt(np.mean(nz ** 2)) + 1e-9
                a = (nz / rms * (10 ** rng.uniform(-3.0, -2.0))).astype(np.float32)   # scale to quiet
        clips.append(a)
    return clips

def silence_activation(model, clips):
    X = np.stack([feat_from_audio(a) for a in clips])
    p = model.predict(X, batch_size=256, verbose=0)[:, 0]
    return float(np.mean(p > 0.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-silence", type=int, default=600)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke: args.epochs, args.n_silence = 2, 30
    C.seed_everything()
    rng = np.random.RandomState(C.SEED)
    sil = make_silence(args.n_silence, rng)
    print(f"[expB] {len(sil)} silence/quiet clips", flush=True)

    order = ["3class", "binary_weights", "binary_oversample", "binary_natural"]
    out = {}
    for key in order:
        model, K, (Xte, yte3) = train(key, args.epochs)
        res = TA.evaluate(model, Xte, yte3, K)
        res["silence_activation"] = round(silence_activation(model, sil), 4)
        res["num_classes"] = K
        out[key] = res
        print(f"  [{key}] AUC={res['wake_auc']:.5f} FAR@2%={res['far_at_frr2pct']:.4f} "
              f"silence_act={res['silence_activation']:.4f}", flush=True)

    final = {"exp": "labelscheme_silence", "epochs": args.epochs, "n_silence": len(sil), "models": out}
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "exp_labelscheme_silence.json")
    json.dump(final, open(p, "w"), indent=2)
    print("\n==== EXP B RESULT ===="); print(json.dumps(final, indent=2)); print("saved ->", p)

if __name__ == "__main__":
    main()

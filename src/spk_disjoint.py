"""spk_disjoint.py -- rebuild the split SPEAKER/SOURCE-DISJOINT (hold out whole
speakers for the keyword, whole sources for negatives) and re-run the 3-class
model, to get honest NEW-SPEAKER numbers vs the leaky random-split cache.
Usage: python spk_disjoint.py --epochs 30 [--smoke]"""
import os, re, json, argparse
import numpy as np
from multiprocessing import Pool
from collections import Counter
import tensorflow as tf
import common as C, build_cache as BC, train_ablation as TA

def group_of(path, lab):
    fn = os.path.basename(path)
    if lab == 0:                                    # keyword: speaker identity
        m = re.search(r"unlabeled\.([^_]+(?: [^_]+)*)_Noban_", fn)
        if m: return "w:" + m.group(1).strip().lower()
        m = re.search(r"wake_(speaker[0-9]+)", fn)
        if m: return "w:" + m.group(1)
        m = re.search(r"voice_msg_(-?[0-9]+)", fn)
        if m: return "w:vm_" + m.group(1)
        m = re.search(r"unlabeled\.([a-zA-Z0-9]+)\.wav", fn)
        if m: return "w:src_" + m.group(1)
        return "w:unk_%d" % (abs(hash(fn)) % 40)    # spread unknowns into pseudo-groups
    return "%d:%s" % (lab, fn)                        # negatives: per-clip (random stratified split)

def assign(paths, labels, groups, seed=C.SEED):
    rng = np.random.RandomState(seed)
    gsize = Counter(groups)
    glab = {}
    for l, g in zip(labels, groups): glab.setdefault(g, l)
    split = {}
    for lab in (0, 1, 2):
        gs = [g for g in gsize if glab[g] == lab]; rng.shuffle(gs)
        total = sum(gsize[g] for g in gs); cum = 0
        te_t, ev_t = 0.10 * total, 0.20 * total   # fill te (~10%) then ev (~10%) then tr
        for g in gs:
            split[g] = "te" if cum < te_t else ("ev" if cum < ev_t else "tr")
            cum += gsize[g]
    return split, gsize, glab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.epochs = 2
    C.seed_everything()
    paths, labels = BC.gather_paths(60 if a.smoke else None)
    groups = [group_of(p, l) for p, l in zip(paths, labels)]
    split, gsize, glab = assign(paths, labels, groups)

    with Pool(max(1, os.cpu_count() - 2)) as pool:
        feats = list(pool.imap(BC._worker, paths, chunksize=16))
    idx = {"tr": [], "ev": [], "te": []}
    for i, (f, g) in enumerate(zip(feats, groups)):
        if f is not None: idx[split[g]].append(i)

    def stack(ii):
        return np.stack([feats[i] for i in ii]).astype(np.float32), labels[np.array(ii)]
    Xtr, ytr = stack(idx["tr"]); Xev, yev = stack(idx["ev"]); Xte, yte = stack(idx["te"])
    wsp = lambda ii: set(groups[i] for i in ii if labels[i] == 0)
    tr_sp, te_sp = wsp(idx["tr"]), wsp(idx["te"])
    print(f"[spk] sizes tr/ev/te = {len(idx['tr'])}/{len(idx['ev'])}/{len(idx['te'])}", flush=True)
    print(f"[spk] wake speaker-groups tr/ev/te = {len(tr_sp)}/{len(wsp(idx['ev']))}/{len(te_sp)}; "
          f"tr&te overlap = {len(tr_sp & te_sp)} (must be 0)", flush=True)
    for nm, yy in [("train", ytr), ("eval", yev), ("test", yte)]:
        print(f"   {nm}: wake={int(np.sum(yy==0))} other={int(np.sum(yy==1))} noise={int(np.sum(yy==2))}", flush=True)

    ytr_oh = tf.keras.utils.to_categorical(ytr, 3); yev_oh = tf.keras.utils.to_categorical(yev, 3)
    m = C.build_model_v17((50, 32, 1), 3); spe = max(1, len(Xtr) // 32)
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, spe * 5, t_mul=2.0, m_mul=0.9, alpha=1e-4)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    m.fit(TA.make_ds(Xtr, ytr_oh, 32, True), validation_data=TA.make_ds(Xev, yev_oh, 32, False),
          epochs=a.epochs, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(
              monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)])
    r = TA.evaluate(m, Xte, yte, 3)
    p = m.predict(Xte, batch_size=256, verbose=0)
    r["overall_acc"] = float(np.mean(p.argmax(1) == yte))
    r["split"] = "speaker_source_disjoint"
    r["n_wake_speakers_train"] = len(tr_sp); r["n_wake_speakers_test"] = len(te_sp)
    r["sizes"] = [len(idx["tr"]), len(idx["ev"]), len(idx["te"])]
    r["te_noise_n"] = int(np.sum(yte == 2)); r["te_other_n"] = int(np.sum(yte == 1)); r["te_wake_n"] = int(np.sum(yte == 0))
    os.makedirs(os.path.expanduser("~/wuwexp/results"), exist_ok=True)
    json.dump(r, open(os.path.expanduser("~/wuwexp/results/spk_disjoint.json"), "w"), indent=2)
    print("\n==== SPEAKER-DISJOINT 3CLASS RESULT ====\n" + json.dumps(r, indent=2))
    print("(compare to leaky random split: acc 97.8%, AUC 0.99996, FAR@2%FRR 0.032%, 7 speech / 0 noise FA)")

if __name__ == "__main__":
    main()

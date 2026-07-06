"""build_spkdisjoint_cache.py -- materialise the SPEAKER/SOURCE-DISJOINT split as
an npz cache (Xtr/ytr/Xev/yev/Xte/yte, 3-class labels wake=0) so that
train_ablation.py can run the 3class/binary/frr comparison on the HONEST split
exactly the way it ran on the leaky random-split cache. Reuses the split logic
from spk_disjoint.py verbatim so the split is identical."""
import os
import numpy as np
from multiprocessing import Pool
import common as C, build_cache as BC
import spk_disjoint as SD

def main():
    C.seed_everything()
    paths, labels = BC.gather_paths(None)
    labels = np.asarray(labels)
    groups = [SD.group_of(p, l) for p, l in zip(paths, labels)]
    split, gsize, glab = SD.assign(paths, labels, groups)

    with Pool(max(1, os.cpu_count() - 2)) as pool:
        feats = list(pool.imap(BC._worker, paths, chunksize=16))
    idx = {"tr": [], "ev": [], "te": []}
    for i, (f, g) in enumerate(zip(feats, groups)):
        if f is not None:
            idx[split[g]].append(i)

    def stack(ii):
        return (np.stack([feats[i] for i in ii]).astype(np.float32),
                labels[np.array(ii)].astype(np.int64))
    Xtr, ytr = stack(idx["tr"]); Xev, yev = stack(idx["ev"]); Xte, yte = stack(idx["te"])

    # sanity: zero speaker overlap between train and test wake groups
    wsp = lambda ii: set(groups[i] for i in ii if labels[i] == 0)
    ov = len(wsp(idx["tr"]) & wsp(idx["te"]))
    print(f"[cache] tr/ev/te = {len(idx['tr'])}/{len(idx['ev'])}/{len(idx['te'])} "
          f"wake-speaker tr&te overlap={ov} (must be 0)", flush=True)
    for nm, yy in [("train", ytr), ("eval", yev), ("test", yte)]:
        print(f"   {nm}: wake={int((yy==0).sum())} other={int((yy==1).sum())} noise={int((yy==2).sum())}", flush=True)

    out = os.path.expanduser("~/wuwexp/cache/features_pcen_spkdisjoint.npz")
    np.savez_compressed(out, Xtr=Xtr, ytr=ytr, Xev=Xev, yev=yev, Xte=Xte, yte=yte)
    print("saved ->", out, Xtr.shape, Xev.shape, Xte.shape, flush=True)

if __name__ == "__main__":
    main()

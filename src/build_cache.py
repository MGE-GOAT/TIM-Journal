"""
build_cache.py — compute streaming-PCEN features for the whole Noban dataset
ONCE and cache to .npz. Every ablation then loads this instantly instead of
recomputing ~46k streaming-PCEN features (the main speed bottleneck).

Deterministic: files are sorted, labels fixed (Wake=0, Other=1, Noise=2), and
the 80/10/10 stratified split uses SEED. All ablations share this exact split.

Usage:
  python build_cache.py --smoke 20        # quick validation, 20 files/class
  python build_cache.py --workers 12      # full build
"""
import os, sys, argparse, glob, time
import numpy as np
from multiprocessing import Pool
import common as C

def gather_paths(per_class=None):
    spec = [(C.WAKE_ROOT, 0), (C.OTHER_ROOT, 1), (C.NOISE_ROOT, 2)]
    paths, labels = [], []
    for root, lab in spec:
        fs = sorted(glob.glob(os.path.join(root, "*.wav")))
        if per_class:
            fs = fs[:per_class]
        paths += fs
        labels += [lab] * len(fs)
    return paths, np.array(labels, dtype=np.int64)

def _worker(path):
    f = C.wav_to_feature(path)      # (50,32,1) or None
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="files per class (0=all)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--logmel", action="store_true", help="plain log-mel features instead of streaming PCEN")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.logmel:
        # switch the (forked) workers' feature front-end to plain log-mel
        C.USE_PCEN = False
        C.LOG_MEL = True
    if args.out is None:
        args.out = os.path.expanduser(
            "~/wuwexp/cache/features_logmel.npz" if args.logmel else "~/wuwexp/cache/features_pcen.npz")

    per = args.smoke if args.smoke > 0 else None
    paths, labels = gather_paths(per)
    print(f"[cache] {len(paths)} files  (Wake={np.sum(labels==0)}, "
          f"Other={np.sum(labels==1)}, Noise={np.sum(labels==2)})  workers={args.workers}")

    t0 = time.time()
    feats = [None] * len(paths)
    with Pool(args.workers) as pool:
        for i, f in enumerate(pool.imap(_worker, paths, chunksize=16)):
            feats[i] = f
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(paths)}  ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
    dt = time.time() - t0

    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep]).astype(np.float32)
    y = labels[keep]
    dropped = len(paths) - len(keep)
    print(f"[cache] extracted {len(keep)} feats in {dt:.0f}s "
          f"({len(keep)/dt:.0f}/s), dropped {dropped}, X={X.shape}")

    # 80/10/10 stratified split, seed-locked (shared by every ablation)
    from sklearn.model_selection import train_test_split
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=C.SEED)
    Xev, Xte, yev, yte = train_test_split(
        Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=C.SEED)
    print(f"[cache] split  train={Xtr.shape} eval={Xev.shape} test={Xte.shape}")
    for name, yy in [("train", ytr), ("eval", yev), ("test", yte)]:
        print(f"   {name}: wake={np.sum(yy==0)} other={np.sum(yy==1)} noise={np.sum(yy==2)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, Xtr=Xtr, ytr=ytr, Xev=Xev, yev=yev, Xte=Xte, yte=yte)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[cache] saved -> {args.out}  ({mb:.1f} MB)")

if __name__ == "__main__":
    main()

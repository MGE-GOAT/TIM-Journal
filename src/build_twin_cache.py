"""build_twin_cache.py -- synthesize the INMP441-domain training corpus by passing
every conventional clip through the measured digital-twin FIR at 16 kHz BEFORE the
(unchanged) PCEN feature front-end. Same paths + SEED => split is bit-identical to
features_pcen.npz, so conventional vs twin arms are directly comparable."""
import os, time, numpy as np
from multiprocessing import Pool
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import common as C, build_cache as BC

FIR = np.load(os.path.expanduser("~/wuwexp/domainshift/twin_fir.npy")).astype(np.float64)
GD = (len(FIR) - 1) // 2                      # linear-phase group delay to compensate
_orig_load = C.load_audio

def _load_twin(path, sr=C.TARGET_SR):
    a = _orig_load(path, sr=sr)
    if a is None:
        return None
    y = lfilter(FIR, 1.0, a)                  # apply the conventional->INMP441 transfer
    y = np.concatenate([y[GD:], np.zeros(GD, dtype=y.dtype)])   # undo the FIR delay
    return y.astype(np.float32)

C.load_audio = _load_twin                     # patched before Pool fork; wav_to_feature picks it up

def main():
    C.seed_everything()
    paths, labels = BC.gather_paths(None)
    labels = np.asarray(labels)
    t0 = time.time()
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        feats = list(pool.imap(BC._worker, paths, chunksize=16))
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep]).astype(np.float32)
    y = labels[keep]
    print("[twin-cache] %d feats in %.0fs  X=%s  (Wake=%d Other=%d Noise=%d)"
          % (len(keep), time.time() - t0, X.shape, (y == 0).sum(), (y == 1).sum(), (y == 2).sum()), flush=True)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=C.SEED)
    Xev, Xte, yev, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=C.SEED)
    out = os.path.expanduser("~/wuwexp/cache/features_pcen_twin.npz")
    np.savez_compressed(out, Xtr=Xtr, ytr=ytr, Xev=Xev, yev=yev, Xte=Xte, yte=yte)
    print("[twin-cache] saved -> %s  tr/ev/te=%d/%d/%d" % (out, len(Xtr), len(Xev), len(Xte)), flush=True)
    # sanity: twin test features should differ from conventional test features on the same split
    conv = np.load(os.path.expanduser("~/wuwexp/cache/features_pcen.npz"))
    d = float(np.mean(np.abs(Xte - conv["Xte"])))
    print("[twin-cache] mean|twin-conv| on test features = %.4f (labels match: %s)"
          % (d, bool(np.array_equal(yte, conv["yte"]))))

if __name__ == "__main__":
    main()

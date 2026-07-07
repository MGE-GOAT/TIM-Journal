import os, glob, json, numpy as np
from scipy.signal import lfilter
from math import comb
import tensorflow as tf
import common as C
FIR = np.load(os.path.expanduser("~/wuwexp/domainshift/twin_fir.npy"))
BASE = "/home/mahrad/storage/Data/wake/WUW-final/Wake Up Word"
negs = sorted(glob.glob(BASE + "/DNC_reconstructed/*.wav"))
print("never-seen negatives:", len(negs), flush=True)
def feat(p):
    a = C.load_audio(p)
    a = lfilter(FIR, 1.0, a).astype(np.float32)   # conventional->INMP441 twin domain (matches build_twin_cache)
    return C.format_ds_cnn(C.extract_ds_cnn_mfe(a))
Xneg = np.stack([feat(p) for p in negs]).astype(np.float32)
d = np.load(os.path.expanduser("~/wuwexp/cache/features_pcen_twin.npz"))
Xte, yte = d["Xte"], d["yte"]; wake = Xte[yte == 0]
def binom1(k, n): return sum(comb(n, i) for i in range(k + 1)) / 2**n
res = {}
for arm in ["A0_conv", "A1_generic", "A3_randtwin", "A2_twin"]:
    m = tf.keras.models.load_model(f"/home/mahrad/wuwexp/models/ladder/{arm}.keras", compile=False)
    thr = float(np.percentile(m.predict(wake, batch_size=256, verbose=0)[:, 0], 2.0))  # 2% FRR
    pn = m.predict(Xneg, batch_size=256, verbose=0)[:, 0]
    fa = int((pn >= thr).sum())
    res[arm] = dict(fa=fa, n=len(negs), far_pct=round(100*fa/len(negs), 3))
    print("%-12s FA %4d / %d = %.3f%%" % (arm, fa, len(negs), 100*fa/len(negs)), flush=True)
a0, a1, a3 = res["A0_conv"]["fa"], res["A1_generic"]["fa"], res["A3_randtwin"]["fa"]
print("\n=== POWERED twin-vs-EQ (the novelty test) ===", flush=True)
print("twin(A3) %d vs generic-EQ(A1) %d : one-sided binomial p=%.4f" % (a3, a1, binom1(min(a3,a1), a3+a1)), flush=True)
print("twin(A3) %d vs conventional(A0) %d : one-sided binomial p=%.4f" % (a3, a0, binom1(min(a3,a0), a3+a0)), flush=True)
res["tests"] = {"twin_vs_eq_p": binom1(min(a3,a1),a3+a1), "twin_vs_conv_p": binom1(min(a3,a0),a3+a0)}
json.dump(res, open(os.path.expanduser("~/wuwexp/results/exp_power.json"), "w"), indent=2)
print("EXP_POWER_DONE", flush=True)

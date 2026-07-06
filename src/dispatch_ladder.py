"""dispatch_ladder.py -- run the ablation ladder across 4 GPU slots (2 on the PC,
2 on this laptop), 2 models per GPU via memory caps. Each job = train_job.py for
one (arm, seed); results are pulled to the laptop and aggregated into a matrix.
Run ON the laptop."""
import os, json, subprocess, threading, queue, time, glob

PC = "mahrad@192.168.1.104"; PORT = "2222"
ENVPY = "~/miniconda3/envs/ml/bin/python"
LD = "$HOME/miniconda3/envs/ml/lib"
WUW = "~/wuwexp"
RES = "~/wuwexp/results/ladder"

ARMS = {                                   # arms runnable offline (A4/A5 need Phase-3 real negs)
    "A0_conv": "~/wuwexp/cache/features_pcen.npz",
    "A2_twin": "~/wuwexp/cache/features_pcen_twin.npz",
}
SEEDS = [99, 7, 21, 42, 123]
SLOTS = [("pc", 3200), ("pc", 3200), ("laptop", 2500), ("laptop", 2500)]

q = queue.Queue()
for arm, cache in ARMS.items():
    for s in SEEDS:
        q.put((arm, cache, s))

def run_job(machine, memmb, arm, cache, seed):
    out = f"{RES}/{arm}_s{seed}.json"
    inner = (f"cd {WUW} && LD_LIBRARY_PATH={LD} TF_CPP_MIN_LOG_LEVEL=2 {ENVPY} train_job.py "
             f"--train-cache '{cache}' --label {arm} --seed {seed} --gpu-mem-mb {memmb} --epochs 25 --out '{out}'")
    cmd = (["ssh", "-p", PORT, "-o", "StrictHostKeyChecking=accept-new", PC, inner]
           if machine == "pc" else ["bash", "-lc", inner])
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = "JOB_DONE" in r.stdout
    print(f"[{machine:6}] {arm} s{seed}: {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("  stdout:", r.stdout[-400:], "\n  stderr:", r.stderr[-400:], flush=True)

def worker(machine, memmb):
    while True:
        try:
            arm, cache, seed = q.get_nowait()
        except queue.Empty:
            return
        run_job(machine, memmb, arm, cache, seed)
        q.task_done()

subprocess.run(["ssh", "-p", PORT, PC, f"mkdir -p {RES}"])
os.makedirs(os.path.expanduser(RES), exist_ok=True)

t0 = time.time()
ts = [threading.Thread(target=worker, args=s) for s in SLOTS]
for t in ts: t.start()
for t in ts: t.join()
print(f"\nall jobs finished in {time.time()-t0:.0f}s", flush=True)

# pull PC result JSONs to the laptop, then aggregate everything
subprocess.run(["bash", "-lc",
    f"scp -P {PORT} {PC}:{RES}/*.json {os.path.expanduser(RES)}/ 2>/dev/null"])
rows = {}
for f in glob.glob(os.path.expanduser(RES) + "/*.json"):
    d = json.load(open(f)); rows.setdefault(d["label"], []).append(d)
import statistics as st
print("\n==== LADDER: TRAIN x TEST MATRIX (mean over seeds) ====")
print("  arm          n   conv-AUC   conv-FAR@2   twin-AUC   twin-FAR@2")
for arm in sorted(rows):
    r = rows[arm]
    def mean(dom, k): return st.mean(x[dom][k] for x in r)
    print("  %-10s  %2d   %.5f   %6.3f%%     %.5f   %6.3f%%" % (
        arm, len(r), mean("conv", "wake_auc"), 100*mean("conv", "far_at_frr2pct"),
        mean("twin", "wake_auc"), 100*mean("twin", "far_at_frr2pct")))
json.dump(rows, open(os.path.expanduser(RES + "/../ladder_agg.json"), "w"), indent=2)
print("LADDER_AGG_DONE")

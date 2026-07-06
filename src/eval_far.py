"""
eval_far.py — streaming false-activation-rate over the ~400h negative corpus.
Faithful port of notebook cell 18 (StreamingPCEN + per-stride PCEN + EMA +
Schmitt re-arm + margin + cooldown), run on an INT8 tflite model, parallelized
across files (each worker = one tflite interpreter, like the device).

Reports false alarms and FA/hour. Use --limit-hours for a fast comparison sweep;
omit for the full corpus (the headline number, matches the thesis 388h stress).

Usage:
  python eval_far.py --tflite models/x_int8.tflite --limit-hours 20 --workers 12
  python eval_far.py --tflite models/x_int8.tflite                  # full corpus
"""
import os, glob, argparse, time, json
import numpy as np
import librosa
from multiprocessing import Pool

BASE = "/home/mahrad/storage/Data/wake/WUW-final/Wake Up Word"
DNC_ROOT   = os.path.join(BASE, "DNC_reconstructed")
OTHER_ROOT = os.path.join(BASE, "all other")

SR=16000; N_FFT=512; HOP=320; LOOKBACK=N_FFT-HOP; N_MELS=32; FMIN=60; FMAX=6000; TW=50
PCEN_ALPHA=0.90; PCEN_DELTA=2.0; PCEN_R=0.5; PCEN_TIME_C=0.10
ALPHA=0.3; TH_UP=0.60; TH_DOWN=0.30; MARGIN=0.20; COOLDOWN=7
STRIDE=int(0.3*SR); WIN=int(1.0*SR); MIN_LEN=int(2.4*SR)

_TFLITE=None; _II=None; _OO=None; _ISC=None; _IZP=None; _OSC=None; _OZP=None

def _warm_zi(n=5):
    sil=np.zeros(WIN,dtype=np.float32)
    m=librosa.feature.melspectrogram(y=sil,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False)
    zi=None
    for _ in range(n):
        _,zi=librosa.pcen(S=m*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
    return zi

_NC=3
def _init(tflite_path):
    global _TFLITE,_II,_OO,_ISC,_IZP,_OSC,_OZP,_NC
    import tensorflow as tf
    _TFLITE=tf.lite.Interpreter(model_path=tflite_path); _TFLITE.allocate_tensors()
    _II=_TFLITE.get_input_details(); _OO=_TFLITE.get_output_details()
    _ISC,_IZP=_II[0]["quantization"]; _OSC,_OZP=_OO[0]["quantization"]
    _NC=int(_OO[0]["shape"][-1])   # 2 (binary/frr) or 3 (3class)

def _infer(feat):
    x=feat/_ISC+_IZP; x=np.clip(x,-128,127).astype(np.int8)
    _TFLITE.set_tensor(_II[0]["index"],x); _TFLITE.invoke()
    o=_TFLITE.get_tensor(_OO[0]["index"])
    return (o.astype(np.float32)-_OZP)*_OSC

def _far_one(path):
    """Return (false_alarms, seconds) for one negative file."""
    try:
        audio,_=librosa.load(path,sr=SR,mono=True)
    except Exception:
        return (0,0.0)
    if audio is None or len(audio)<MIN_LEN:
        return (0,0.0)
    secs=len(audio)/SR
    zi=_warm_zi(); mel_buf=np.zeros((TW,N_MELS),dtype=np.float32); lookback=np.zeros(LOOKBACK,dtype=np.float32)
    ema=np.zeros(_NC,dtype=np.float32); cooldown=0; armed=True; fa=0
    for s in range(0,len(audio)-STRIDE+1,STRIDE):
        chunk=audio[s:s+STRIDE].astype(np.float32)
        inp=np.concatenate([lookback,chunk])
        mel=librosa.feature.melspectrogram(y=inp,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False)
        if len(chunk)>=LOOKBACK: lookback=chunk[-LOOKBACK:].astype(np.float32).copy()
        else: lookback=np.concatenate([lookback,chunk])[-LOOKBACK:].astype(np.float32)
        mel,zi=librosa.pcen(S=mel*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
        nf=mel.T; n_new=nf.shape[0]
        if n_new>=TW: mel_buf=nf[-TW:].astype(np.float32)
        elif n_new>0:
            mel_buf=np.roll(mel_buf,-n_new,axis=0); mel_buf[-n_new:]=nf
        feat=mel_buf[np.newaxis,...,np.newaxis].astype(np.float32)
        preds=_infer(feat)[0]
        ema=ALPHA*preds+(1.0-ALPHA)*ema
        ew=float(ema[0]); margin=ew-float(np.max(ema[1:]))
        if not armed and ew<TH_DOWN: armed=True
        if cooldown>0: cooldown-=1
        if cooldown==0 and armed and ew>TH_UP and margin>MARGIN:
            fa+=1; armed=False; cooldown=COOLDOWN
    return (fa,secs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tflite",required=True)
    ap.add_argument("--limit-hours",type=float,default=0.0,help="0=full corpus")
    ap.add_argument("--workers",type=int,default=max(1,os.cpu_count()-2))
    ap.add_argument("--tag",default="")
    args=ap.parse_args()
    files=sorted(glob.glob(os.path.join(DNC_ROOT,"*.wav")))+sorted(glob.glob(os.path.join(OTHER_ROOT,"*.wav")))
    print(f"[far] corpus: {len(files)} files  model={os.path.basename(args.tflite)}  workers={args.workers}")
    if args.limit_hours>0:
        # ~ estimate: take a prefix; refine by accumulating until target hours
        files=files[:int(args.limit_hours*3600/3)+2000]   # rough; actual hours reported below
        print(f"[far] SUBSET: first {len(files)} files (~{args.limit_hours}h target)")
    t0=time.time(); tot_fa=0; tot_s=0.0; done=0
    with Pool(args.workers,initializer=_init,initargs=(args.tflite,)) as pool:
        for fa,secs in pool.imap_unordered(_far_one,files,chunksize=8):
            tot_fa+=fa; tot_s+=secs; done+=1
            if args.limit_hours>0 and tot_s>=args.limit_hours*3600: pass
            if done%5000==0:
                print(f"  {done}/{len(files)}  {tot_s/3600:.1f}h  FA={tot_fa}  ({time.time()-t0:.0f}s)",flush=True)
    hours=tot_s/3600
    res={"tflite":os.path.basename(args.tflite),"files":done,"hours":round(hours,2),
         "false_alarms":tot_fa,"fa_per_hour":round(tot_fa/hours,4) if hours>0 else None,
         "wall_s":round(time.time()-t0,1)}
    print("\n==== FAR RESULT ===="); print(json.dumps(res,indent=2))
    out=os.path.expanduser(f"~/wuwexp/results/far_{os.path.splitext(os.path.basename(args.tflite))[0]}{('_'+args.tag) if args.tag else ''}.json")
    json.dump(res,open(out,"w"),indent=2); print("saved ->",out)

if __name__=="__main__":
    main()

"""
analyze_recording.py — stream ONE recording through the 3-class int8 model with
the exact deployment EMA/Schmitt/margin/cooldown logic, and report wake fires
(count + timestamps). Used for the domain-shift A/B: run on the reference-mic
recording vs the INMP441 recording of the SAME played audio.

  --mode ref  : reference mic (S16) -> librosa normalizes to [-1,1]
  --mode inmp : INMP441 (2ch S32)   -> LEFT channel (ch0) * (1/2**MIC_SCALE_POW),
                exactly what the device feeds the model (main.cpp micleft path)

Usage:
  python analyze_recording.py --wav rec.wav --mode inmp --tflite models/3class_qat_final_int8.tflite
"""
import argparse, os, json
import numpy as np, librosa, soundfile as sf, tensorflow as tf

SR=16000; N_FFT=512; HOP=320; LOOKBACK=N_FFT-HOP; N_MELS=32; FMIN=60; FMAX=6000; TW=50
PCEN_ALPHA=0.90; PCEN_DELTA=2.0; PCEN_R=0.5; PCEN_TIME_C=0.10
ALPHA=0.3; TH_UP=0.60; TH_DOWN=0.30; MARGIN=0.20
# device flow: on a fire it records the command for 5 s, then resumes listening
# -> a 5 s "deaf" window after each fire. 5 s / 0.3 s stride ~= 17 frames.
COOLDOWN=17
STRIDE=int(0.3*SR); WIN=int(1.0*SR)

def load_audio(path, mode, mic_scale_pow=27):
    if mode == "inmp":
        data, sr = sf.read(path, dtype="int32", always_2d=True)   # 2ch S32
        x = data[:, 0].astype(np.float64) / float(2**mic_scale_pow)  # LEFT ch, device scale
        if sr != SR:
            x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR)
        return x.astype(np.float32)
    # reference: normal normalized load
    x, _ = librosa.load(path, sr=SR, mono=True)
    return x.astype(np.float32)

def warm_zi(n=5):
    sil=np.zeros(WIN,dtype=np.float32)
    m=librosa.feature.melspectrogram(y=sil,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False)
    zi=None
    for _ in range(n):
        _,zi=librosa.pcen(S=m*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
    return zi

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wav",required=True); ap.add_argument("--mode",required=True,choices=["ref","inmp"])
    ap.add_argument("--tflite",required=True); ap.add_argument("--pow",type=int,default=27)
    args=ap.parse_args()

    audio=load_audio(args.wav,args.mode,args.pow)
    it=tf.lite.Interpreter(model_path=args.tflite); it.allocate_tensors()
    ii=it.get_input_details()[0]; oo=it.get_output_details()[0]
    isc,izp=ii["quantization"]; osc,ozp=oo["quantization"]; nc=int(oo["shape"][-1])
    def infer(feat):
        x=np.clip(feat/isc+izp,-128,127).astype(np.int8)
        it.set_tensor(ii["index"],x); it.invoke()
        return (it.get_tensor(oo["index"]).astype(np.float32)-ozp)*osc

    zi=warm_zi(); mel_buf=np.zeros((TW,N_MELS),dtype=np.float32); lookback=np.zeros(LOOKBACK,dtype=np.float32)
    ema=np.zeros(nc,dtype=np.float32); cooldown=0; armed=True; fires=[]
    peak=float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms=float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    for k,s in enumerate(range(0,len(audio)-STRIDE+1,STRIDE)):
        chunk=audio[s:s+STRIDE].astype(np.float32)
        inp=np.concatenate([lookback,chunk])
        mel=librosa.feature.melspectrogram(y=inp,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=1.0,center=False)
        if len(chunk)>=LOOKBACK: lookback=chunk[-LOOKBACK:].astype(np.float32).copy()
        else: lookback=np.concatenate([lookback,chunk])[-LOOKBACK:].astype(np.float32)
        mel,zi=librosa.pcen(S=mel*(2**31),sr=SR,hop_length=HOP,time_constant=PCEN_TIME_C,gain=PCEN_ALPHA,bias=PCEN_DELTA,power=PCEN_R,eps=1e-3,zi=zi,return_zf=True)
        nf=mel.T; n_new=nf.shape[0]
        if n_new>=TW: mel_buf=nf[-TW:].astype(np.float32)
        elif n_new>0: mel_buf=np.roll(mel_buf,-n_new,axis=0); mel_buf[-n_new:]=nf
        feat=mel_buf[np.newaxis,...,np.newaxis].astype(np.float32)
        ema=ALPHA*infer(feat)[0]+(1.0-ALPHA)*ema
        ew=float(ema[0]); m=ew-float(np.max(ema[1:]))
        if not armed and ew<TH_DOWN: armed=True
        if cooldown>0: cooldown-=1
        if cooldown==0 and armed and ew>TH_UP and m>MARGIN:
            fires.append(round(s/SR,2)); armed=False; cooldown=COOLDOWN
    res={"wav":os.path.basename(args.wav),"mode":args.mode,"dur_s":round(len(audio)/SR,1),
         "peak":round(peak,4),"rms":round(rms,5),"fires":len(fires),"fire_times_s":fires}
    print(json.dumps(res,indent=2))

if __name__=="__main__":
    main()

"""
common.py — faithful port of the Ds-CNN-QAT notebook's feature + model code.
Reused unchanged across all ablations so features/model are bit-identical to
the original training. (Ported from cells 0, 1, 2, 6.)
"""
import os, random
import numpy as np
import librosa

# ============================ Config (cell 0) ============================
# Data root: the notebook used relative "Noban/{Wake,Other,Noise}". We point at
# the real dir on the PC (spaces are fine inside a Python string).
BASE_DIR   = "/home/mahrad/storage/Data/wake/WUW-final/Wake Up Word/1"
NOBAN_ROOT = os.path.join(BASE_DIR, "Noban")
WAKE_ROOT  = os.path.join(NOBAN_ROOT, "Wake")
OTHER_ROOT = os.path.join(NOBAN_ROOT, "Other")
NOISE_ROOT = os.path.join(NOBAN_ROOT, "Noise")

TARGET_SR  = 16000
TARGET_LEN = 1 * TARGET_SR

N_MELS       = 32
N_FFT        = 512
HOP_LENGTH   = 320
LOOKBACK     = N_FFT - HOP_LENGTH   # 192
FMIN         = 60
FMAX         = 6000
TARGET_WIDTH = 50

USE_PCEN     = True
NORMALIZE    = 'per_sample'
LOG_MEL      = False

PCEN_ALPHA   = 0.90
PCEN_DELTA   = 2.0
PCEN_R       = 0.5
PCEN_TIME_C  = 0.10

STRIDE_LEN    = int(0.3 * TARGET_SR)   # 4800
STREAM_LEN    = 8 * STRIDE_LEN         # 38400 = 2.4 s

SEED = 99

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf; tf.random.set_seed(seed)
    except Exception:
        pass

# ============================ Audio loading (cell 1) ============================
def load_audio(path, sr=TARGET_SR):
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    if audio is None or len(audio) == 0:
        return None
    if len(audio) < STREAM_LEN:
        audio = np.pad(audio, (0, STREAM_LEN - len(audio)))
    elif len(audio) > STREAM_LEN:
        audio = audio[:STREAM_LEN]
    if np.isnan(audio).any():
        return None
    return audio.astype(np.float32)

# ============================ Streaming PCEN features (cell 2) ============================
def _make_pcen_zi_init(time_c=PCEN_TIME_C, n_warmup=5):
    silence = np.zeros(TARGET_LEN, dtype=np.float32)
    mel_sil = librosa.feature.melspectrogram(
        y=silence, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0, center=False)
    zi = None
    for _ in range(n_warmup):
        _, zi = librosa.pcen(S=mel_sil * (2**31), sr=TARGET_SR, hop_length=HOP_LENGTH,
            time_constant=time_c, gain=PCEN_ALPHA, bias=PCEN_DELTA, power=PCEN_R,
            eps=1e-3, zi=zi, return_zf=True)
    return zi

PCEN_ZI_INIT = _make_pcen_zi_init()

def _stream_pcen_into_buffer(audio, zi, time_c, mel_buffer):
    cursor = 0; n = len(audio)
    lookback = np.zeros(LOOKBACK, dtype=np.float32)
    while cursor < n:
        end = min(cursor + STRIDE_LEN, n)
        chunk = audio[cursor:end]; cursor = end
        input_audio = np.concatenate([lookback, chunk])
        if len(input_audio) < N_FFT:
            break
        mel = librosa.feature.melspectrogram(
            y=input_audio, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0, center=False)
        mel, zi = librosa.pcen(S=mel * (2**31), sr=TARGET_SR, hop_length=HOP_LENGTH,
            time_constant=time_c, gain=PCEN_ALPHA, bias=PCEN_DELTA, power=PCEN_R,
            eps=1e-3, zi=zi, return_zf=True)
        if len(chunk) >= LOOKBACK:
            lookback = chunk[-LOOKBACK:].astype(np.float32).copy()
        else:
            lookback = np.concatenate([lookback, chunk])[-LOOKBACK:].astype(np.float32)
        new_frames = mel.T
        n_new = new_frames.shape[0]
        if n_new == 0:
            continue
        if n_new >= TARGET_WIDTH:
            mel_buffer = new_frames[-TARGET_WIDTH:].astype(np.float32)
        else:
            mel_buffer = np.roll(mel_buffer, -n_new, axis=0)
            mel_buffer[-n_new:] = new_frames
    return mel_buffer, zi

def extract_ds_cnn_mfe(audio):
    if USE_PCEN:
        zi = PCEN_ZI_INIT.copy()
        mel_buffer = np.zeros((TARGET_WIDTH, N_MELS), dtype=np.float32)
        mel_buffer, _ = _stream_pcen_into_buffer(audio, zi, PCEN_TIME_C, mel_buffer)
        mel = mel_buffer
    else:
        target = audio[-TARGET_LEN:]
        mel = librosa.feature.melspectrogram(
            y=target, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0, center=False)
        if LOG_MEL:
            mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        mel = mel.T
    if mel.shape[0] < TARGET_WIDTH:
        mel = np.pad(mel, ((TARGET_WIDTH - mel.shape[0], 0), (0, 0)))
    else:
        mel = mel[-TARGET_WIDTH:]
    if not USE_PCEN:
        mean, std = np.mean(mel), np.std(mel)
        mel = (mel - mean) / (std + 1e-9)
    return mel.astype(np.float32)

def format_ds_cnn(mfe):
    return mfe[..., np.newaxis]

def wav_to_feature(path):
    """path -> (50,32,1) feature or None."""
    a = load_audio(path)
    if a is None:
        return None
    return format_ds_cnn(extract_ds_cnn_mfe(a))

# ============================ Model: Noban V17 (cell 6) ============================
def coord_attention(x, T_eff, F_eff, reduction=4):
    from tensorflow.keras import layers
    ch = x.shape[-1]; r = max(1, ch // reduction)
    t = layers.AveragePooling2D((1, F_eff))(x)
    t = layers.Conv2D(r, 1, use_bias=False)(t); t = layers.BatchNormalization()(t); t = layers.ReLU()(t)
    t_logit = layers.Conv2D(ch, 1, use_bias=False)(t)
    f = layers.AveragePooling2D((T_eff, 1))(x)
    f = layers.Conv2D(r, 1, use_bias=False)(f); f = layers.BatchNormalization()(f); f = layers.ReLU()(f)
    f_logit = layers.Conv2D(ch, 1, use_bias=False)(f)
    gate = layers.Activation('sigmoid')(layers.Add()([t_logit, f_logit]))
    return layers.Multiply()([x, gate])

def se_block(x, reduction=4):
    from tensorflow.keras import layers
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(keepdims=True)(x)
    se = layers.Conv2D(max(1, channels // reduction), 1, use_bias=False)(se); se = layers.ReLU()(se)
    se = layers.Conv2D(channels, 1, use_bias=False)(se); se = layers.Activation('sigmoid')(se)
    return layers.Multiply()([x, se])

def build_model_v17(input_shape=(50, 32, 1), num_classes=3):
    import tensorflow as tf
    from tensorflow.keras import layers
    T, F_bins = input_shape[0], input_shape[1]
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (1, 4), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(32, (1, 6), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(32, (6, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    dilation_schedule = [(1, 1), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3)]
    channel_schedule  = [32, 48, 48, 64, 64, 64]
    T_eff, F_eff = T, F_bins
    for i, ((dil_t, dil_f), ch) in enumerate(zip(dilation_schedule, channel_schedule)):
        shortcut = x
        x_dw = layers.DepthwiseConv2D((5, 3), padding='same', dilation_rate=(dil_t, dil_f), use_bias=False)(x)
        x_dw = layers.BatchNormalization()(x_dw); x_dw = layers.ReLU()(x_dw)
        if i >= 3:
            x_dw = coord_attention(x_dw, T_eff, F_eff)
        x = layers.Conv2D(ch, (1, 1), padding='same', use_bias=False)(x_dw)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != ch:
            shortcut = layers.Conv2D(ch, (1, 1), padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.Add()([x, shortcut]); x = se_block(x); x = layers.ReLU()(x)
    x = coord_attention(x, T_eff, F_eff)
    x = layers.DepthwiseConv2D((1, F_eff), padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.DepthwiseConv2D((T_eff, 1), padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, use_bias=True)(x)
    outputs = layers.Activation('softmax')(x)
    return tf.keras.Model(inputs, outputs, name="Noban_V17")

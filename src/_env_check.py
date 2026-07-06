import tensorflow as tf, librosa, numpy as np, sklearn, sys
gpus = tf.config.list_physical_devices('GPU')
print("python:", sys.version.split()[0])
print("tf:", tf.__version__, "| librosa:", librosa.__version__, "| numpy:", np.__version__)
print("GPUs visible to TF:", [g.name for g in gpus] or "NONE (CPU)")

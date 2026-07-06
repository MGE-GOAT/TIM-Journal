"""tflite_utils.py — full-int8 TFLite conversion (matches notebook cells 13-14).
Representative data = a balanced subset of cached training features."""
import numpy as np
import tensorflow as tf

def to_int8_tflite(model, X_rep, out_path, n_rep=300):
    idx = np.random.RandomState(0).choice(len(X_rep), size=min(n_rep, len(X_rep)), replace=False)
    rep = X_rep[idx].astype(np.float32)
    def rep_ds():
        for i in range(len(rep)):
            yield [rep[i:i+1]]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_ds
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    with open(out_path, "wb") as f:
        f.write(blob)
    return len(blob)

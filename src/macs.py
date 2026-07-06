"""macs.py — compute MACs (and params) for Noban-V17 + baselines, for the
efficiency table. FLOPs via the TF profiler; MACs = FLOPs/2."""
import json, os
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import common as C
from models_baseline import REGISTRY

def flops_of(model):
    ishape = [1] + list(model.input_shape[1:])
    conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec(ishape, tf.float32))
    frozen = convert_variables_to_constants_v2(conc)
    opts = ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).with_empty_output().build()
    info = profile(frozen.graph, options=opts)
    return info.total_float_ops

def main():
    rows = []
    models = {"noban_v17": lambda nc: C.build_model_v17(num_classes=nc)}
    models.update(REGISTRY)
    for name, fn in models.items():
        m = fn(3)
        p = int(m.count_params())
        try:
            fl = flops_of(m)
            macs = fl // 2
        except Exception as e:
            fl, macs = None, None
            print(f"  [{name}] flops failed: {e}")
        rows.append({"model": name, "params": p, "flops": fl, "macs": macs})
        print(f"  {name:12s} params={p:>9,}  MACs={('%.1fM'%(macs/1e6)) if macs else 'NA'}")
        tf.keras.backend.clear_session()
    out = os.path.expanduser("~/wuwexp/results/macs.json")
    json.dump(rows, open(out, "w"), indent=2)
    print("saved ->", out)

if __name__ == "__main__":
    main()

"""
models_baseline.py — standard lightweight-KWS baselines for the "compared to
what" table, all on the same (50,32,1) PCEN input, 3-class head.

  * dscnn_s / dscnn_m : DS-CNN (Zhang et al. 2017, "Hello Edge") — THE canonical
                        depthwise-separable KWS baseline the Noban model derives
                        from. Small and medium widths.
  * cnn_trad          : classic 2-conv KWS CNN (Sainath & Parada 2015 style) —
                        a second, non-DS reference point.
  * tc_resnet8        : TC-ResNet8 (Choi et al. 2019) — temporal 1D-style resnet,
                        another widely-cited tiny-KWS baseline.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def _sep_block(x, ch, k=(3, 3)):
    x = layers.DepthwiseConv2D(k, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(ch, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    return x


def build_dscnn(input_shape=(50, 32, 1), num_classes=3, width=64, n_blocks=4):
    inp = tf.keras.Input(input_shape)
    x = layers.Conv2D(width, (10, 4), strides=(2, 2), padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    for _ in range(n_blocks):
        x = _sep_block(x, width)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out, name=f"DSCNN_w{width}_b{n_blocks}")


def build_cnn_trad(input_shape=(50, 32, 1), num_classes=3):
    inp = tf.keras.Input(input_shape)
    x = layers.Conv2D(64, (8, 4), padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (4, 4), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out, name="CNN_trad")


def _tc_block(x, ch, stride=1):
    sc = x
    x = layers.Conv2D(ch, (9, 1), strides=(stride, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(ch, (9, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if sc.shape[-1] != ch or stride != 1:
        sc = layers.Conv2D(ch, 1, strides=(stride, 1), padding='same', use_bias=False)(sc)
        sc = layers.BatchNormalization()(sc)
    x = layers.Add()([x, sc]); x = layers.ReLU()(x)
    return x


def build_tc_resnet8(input_shape=(50, 32, 1), num_classes=3, k=1.0):
    # treat freq as channels: (T, F, 1) -> (T, 1, F)
    inp = tf.keras.Input(input_shape)
    x = layers.Permute((1, 3, 2))(inp)                    # (T,1,F)
    x = layers.Conv2D(int(16 * k), (3, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = _tc_block(x, int(24 * k), stride=2)
    x = _tc_block(x, int(32 * k), stride=2)
    x = _tc_block(x, int(48 * k), stride=2)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out, name="TC_ResNet8")


REGISTRY = {
    "dscnn_s":     lambda nc: build_dscnn(num_classes=nc, width=48, n_blocks=4),
    "dscnn_m":     lambda nc: build_dscnn(num_classes=nc, width=64, n_blocks=5),
    "cnn_trad":    lambda nc: build_cnn_trad(num_classes=nc),
    "tc_resnet8":  lambda nc: build_tc_resnet8(num_classes=nc),
}

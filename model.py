import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPooling2D, UpSampling2D,
    BatchNormalization, Activation, Add, Concatenate, Dropout,
    GlobalAveragePooling2D, Reshape, Multiply
)
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# ================================


# ============================================================
# FACTORIES (NO TOPOLOGY CHANGE; only replace layer TYPES)
# ============================================================
def _make_norm(norm_type: str):
    nt = str(norm_type).lower()
    if nt in ["layer", "ln"]:
        return LayerNormalization
    if nt in ["group", "gn"] and GroupNormalization is not None:
        return lambda **kw: GroupNormalization(groups=8, axis=-1, **kw)
    return BatchNormalization

def _make_conv_for_conv2d(conv_type: str):
    ct = str(conv_type).lower()
    if ct in ["sep", "separable", "separableconv2d"]:
        return SeparableConv2D
    return Conv2D

def _make_pool(pooling_type: str):
    pt = str(pooling_type).lower()
    if pt in ["avg", "average", "averagepool", "averagepool2d"]:
        return AveragePooling2D
    return MaxPooling2D

def _fuse(fusion_method: str, name: str, tensors):
    fm = str(fusion_method).lower()
    if fm in ["mul", "multiply"]:
        return Multiply(name=name)(tensors)
    return Add(name=name)(tensors)

def _act(x, activation: str, name: str):
    return Activation(str(activation).lower(), name=name)(x)


def _make_filters(base_filters):
    f = int(base_filters)
    if f < 8:
        f = 8
    elif f > 256:
        f = 256
    return (f // 8) * 8

# ============================================================
# ✅ SINGLE CANONICAL WeightedSum2 (TOP-LEVEL)
# ============================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class WeightedSum2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(2,),
            initializer="ones",
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs):
        a, b = inputs
        out_dtype = a.dtype

        w = tf.cast(self.w, tf.float32)
        w_pos = tf.nn.relu(w)
        w_norm = w_pos / (tf.reduce_sum(w_pos) + tf.constant(1e-6, tf.float32))

        a32 = tf.cast(a, tf.float32)
        b32 = tf.cast(b, tf.float32)
        y32 = w_norm[0] * a32 + w_norm[1] * b32
        return tf.cast(y32, out_dtype)

    def get_config(self):
        return super().get_config()
        
        
# ============================================================
# MODULE 1: Agra  (structure unchanged)
# ============================================================
def Agra(x, filters, dropout=0.01, name="blk",
         kernel_size=3, activation="relu", norm_type="batch",
         conv_type="conv2d", fusion_method="add"):
    Norm  = _make_norm(norm_type)
    ConvK = _make_conv_for_conv2d(conv_type)

    shortcut = x

    x = SeparableConv2D(filters, kernel_size, padding="same", use_bias=False, name=f"{name}_sep1")(x)
    x = Norm(name=f"{name}_bn1")(x)
    x = _act(x, activation, name=f"{name}_act1")

    if dropout and dropout > 0:
        x = Dropout(dropout, name=f"{name}_drop")(x)

    x = SeparableConv2D(filters, kernel_size, padding="same", use_bias=False, name=f"{name}_sep2")(x)
    x = Norm(name=f"{name}_bn2")(x)

    if shortcut.shape[-1] != filters:
        shortcut = ConvK(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = Norm(name=f"{name}_proj_bn")(shortcut)

    x = _fuse(fusion_method, name=f"{name}_fuse", tensors=[x, shortcut])
    x = _act(x, activation, name=f"{name}_out")
    return x

# ============================================================
# MODULE 2: Rota (structure unchanged)
# ============================================================
def Rota(x, filters, rates=(1, 2, 4), dropout=0.01, name="msc",
         kernel_size=3, activation="relu", norm_type="batch",
         conv_type="conv2d", fusion_method="add"):
    Norm  = _make_norm(norm_type)
    ConvK = _make_conv_for_conv2d(conv_type)

    def sep_bn_act(inp, f, k=3, d=1, dr=0.0, nm="s"):
        y = SeparableConv2D(f, k, padding="same", dilation_rate=d, use_bias=False, name=f"{nm}_sep")(inp)
        y = Norm(name=f"{nm}_bn")(y)
        y = _act(y, activation, name=f"{nm}_act")
        if dr and dr > 0:
            y = Dropout(dr, name=f"{nm}_drop")(y)
        return y

    in_ch = int(x.shape[-1])

    b0 = sep_bn_act(x, filters, k=kernel_size, d=rates[0], dr=dropout, nm=f"{name}_b0_r{rates[0]}")
    fused = b0
    for i, r in enumerate(rates[1:], start=1):
        bi = sep_bn_act(x, filters, k=kernel_size, d=r, dr=dropout, nm=f"{name}_b{i}_r{r}")
        fused = _fuse(fusion_method, name=f"{name}_sum{i}", tensors=[fused, bi])

    g = GlobalAveragePooling2D(name=f"{name}_gap")(fused)
    g = Reshape((1, 1, filters), name=f"{name}_greshape")(g)

    g = ConvK(filters, 1, padding="same", activation="sigmoid", name=f"{name}_gate1x1")(g)
    fused = Multiply(name=f"{name}_gated")([fused, g])

    fused = ConvK(filters, 1, padding="same", use_bias=False, name=f"{name}_fuse1x1")(fused)
    fused = Norm(name=f"{name}_fuse_bn")(fused)
    fused = _act(fused, activation, name=f"{name}_fuse_act")

    if in_ch != filters:
        skip = ConvK(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(x)
        skip = Norm(name=f"{name}_proj_bn")(skip)
    else:
        skip = x

    out = _fuse(fusion_method, name=f"{name}_out_fuse", tensors=[fused, skip])
    out = _act(out, activation, name=f"{name}_out_act")
    return out

# ============================================================
# MODULE 3: Petra (structure unchanged)
# ============================================================
def Petra(a, b, filters, dropout=0.01, name="awf",
          kernel_size=3, activation="relu", norm_type="batch",
          conv_type="conv2d", fusion_method="add"):
    Norm  = _make_norm(norm_type)
    ConvK = _make_conv_for_conv2d(conv_type)

    def sep_bn_act(inp, f, k=3, d=1, dr=0.0, nm="s"):
        y = SeparableConv2D(f, k, padding="same", dilation_rate=d, use_bias=False, name=f"{nm}_sep")(inp)
        y = Norm(name=f"{nm}_bn")(y)
        y = _act(y, activation, name=f"{nm}_act")
        if dr and dr > 0:
            y = Dropout(dr, name=f"{nm}_drop")(y)
        return y

    a1 = ConvK(filters, 1, padding="same", use_bias=False, name=f"{name}_a1x1")(a)
    a1 = Norm(name=f"{name}_a_bn")(a1)

    b1 = ConvK(filters, 1, padding="same", use_bias=False, name=f"{name}_b1x1")(b)
    b1 = Norm(name=f"{name}_b_bn")(b1)

    fused = WeightedSum2(name=f"{name}_wsum")([a1, b1])

    g = Concatenate(name=f"{name}_gcat")([a1, b1])
    g = ConvK(1, 1, padding="same", activation="sigmoid", name=f"{name}_gate")(g)
    fused = Multiply(name=f"{name}_gated")([fused, g])

    fused = sep_bn_act(fused, filters, k=kernel_size, d=1, dr=dropout, nm=f"{name}_ref")
    return fused

# ============================================================
# Model (structure unchanged)
# ============================================================
def model_arch(input_shape=(256, 256, 1), num_classes=4, base_filters=32,
                        kernel_size=3, activation="relu", norm_type="batch",
                        conv_type="conv2d", pooling_type="maxpool2d", fusion_method="add", dropout_deep=0.2, multilabel=False,):

    Norm  = _make_norm(norm_type)
    ConvK = _make_conv_for_conv2d(conv_type)
    Pool  = _make_pool(pooling_type)

    inputs = Input(input_shape, name="input")

    c1 = Agra(inputs, base_filters, dropout=0.01, name="enc1",
              kernel_size=kernel_size, activation=activation, norm_type=norm_type,
              conv_type=conv_type, fusion_method=fusion_method)
    p1 = Pool((2, 2), name="pool1")(c1)

    c2 = Agra(p1, base_filters * 2, dropout=0.01, name="enc2",
              kernel_size=kernel_size, activation=activation, norm_type=norm_type,
              conv_type=conv_type, fusion_method=fusion_method)
    p2 = Pool((2, 2), name="pool2")(c2)

    c3 = Agra(p2, base_filters * 4, dropout=dropout_deep, name="enc3",
              kernel_size=kernel_size, activation=activation, norm_type=norm_type,
              conv_type=conv_type, fusion_method=fusion_method)
    p3 = Pool((2, 2), name="pool3")(c3)

    b1 = Rota(p3, base_filters * 8, rates=(1, 2, 4), dropout=dropout_deep, name="msc_bn",
              kernel_size=kernel_size, activation=activation, norm_type=norm_type,
              conv_type=conv_type, fusion_method=fusion_method)

    u1 = UpSampling2D((2, 2), interpolation="bilinear", name="up1")(b1)
    u1 = ConvK(base_filters * 4, 1, padding="same", use_bias=False, name="up1_conv")(u1)
    u1 = Norm(name="up1_bn")(u1)
    u1 = _act(u1, activation, name="up1_act")
    u1 = Petra(u1, c3, filters=base_filters * 4, dropout=dropout_deep, name="awf1",
               kernel_size=kernel_size, activation=activation, norm_type=norm_type,
               conv_type=conv_type, fusion_method=fusion_method)
    c4 = Agra(u1, base_filters * 4, dropout=dropout_deep, name="dec1",
              kernel_size=kernel_size, activation=activation, norm_type=norm_type,
              conv_type=conv_type, fusion_method=fusion_method)

    u2 = UpSampling2D((2, 2), interpolation="bilinear", name="up2")(c4)
    u2 = ConvK(base_filters * 2, 1, padding="same", use_bias=False, name="up2_conv")(u2)
    u2 = Norm(name="up2_bn")(u2)
    u2 = _act(u2, activation, name="up2_act")
    u2 = Petra(u2, c2, filters=base_filters * 2, dropout=0.01, name="awf2",
               kernel_size=kernel_size, activation=activation, norm_type=norm_type,
               conv_type=conv_type, fusion_method=fusion_method)
    c5 = Agra(u2, base_filters * 2, dropout=0.01, name="dec2",
              kernel_size=kernel_size, activation=activation, norm_type=norm_type,
              conv_type=conv_type, fusion_method=fusion_method)

    u3 = UpSampling2D((2, 2), interpolation="bilinear", name="up3")(c5)
    u3 = ConvK(base_filters, 1, padding="same", use_bias=False, name="up3_conv")(u3)
    u3 = Norm(name="up3_bn")(u3)
    u3 = _act(u3, activation, name="up3_act")
    u3 = Petra(u3, c1, filters=base_filters, dropout=0.01, name="awf3",
               kernel_size=kernel_size, activation=activation, norm_type=norm_type,
               conv_type=conv_type, fusion_method=fusion_method)
    c6 = Agra(u3, base_filters, dropout=0.01, name="dec3",
              kernel_size=kernel_size, activation=activation, norm_type=norm_type,
              conv_type=conv_type, fusion_method=fusion_method)

    act_last = "sigmoid" if multilabel else "softmax"
    outputs = ConvK(num_classes, 1, activation=act_last, dtype="float32", name="out")(c6)
    return Model(inputs, outputs, name="ELPS_Net")

# ============================================================
# GFLOPs
# ============================================================
def compute_gflops(model, input_shape):
    @tf.function
    def f(x): return model(x)
    concrete = f.get_concrete_function(tf.TensorSpec([1,*input_shape],tf.float32))
    frozen = convert_variables_to_constants_v2(concrete)

    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(frozen.graph.as_graph_def(), name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(g, options=opts)
    return flops.total_float_ops/1e9

INPUT_SHAPE = (256,256,1)
   
model = model_arch(input_shape=(256, 256, 1), num_classes=4, base_filters=32,
                        kernel_size=3, activation="relu", norm_type="batch",
                        conv_type="separableconv2d", pooling_type="maxpool2d", fusion_method="add", dropout_deep=0.2, multilabel=False,)

print(f"Params: {model.count_params()/1e6:.3f} M | GFLOPs: {compute_gflops(model, INPUT_SHAPE):.3f}")
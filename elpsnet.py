import os, json, math, random, time, copy, contextlib, io, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D,
    MaxPooling2D, AveragePooling2D, UpSampling2D,
    BatchNormalization, LayerNormalization,
    Activation, Add, Dropout,
    GlobalAveragePooling2D, Reshape, Multiply, Concatenate
)
from tensorflow.keras.models import Model

# ---------------- TF spam down ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ---------------- AMP (RTX 2080 Ti) ----------------
mixed_precision.set_global_policy("mixed_float16")
print("Policy:", mixed_precision.global_policy())

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---------------- Optional GroupNorm ----------------
try:
    from tensorflow_addons.layers import GroupNormalization
except Exception:
    GroupNormalization = None

# ============================================================
# REQUIRED USER DEFINITIONS CHECK (must exist ABOVE)
# ============================================================
_req = ["ce_tversky_multiclass", "MeanDiceForeground", "MeanIoUForeground", "MeanHD95Foreground"]
_missing = [k for k in _req if k not in globals()]
if _missing:
    raise RuntimeError(
        "Missing required definitions in this notebook cell context:\n"
        + "\n".join([f"- {k}" for k in _missing]) +
        "\n\nPaste your METRICS + LOSSES cell ABOVE this ENAS cell."
    )

# ============================================================
# USER SWITCHES
# ============================================================
RESUME_ENAS = False
DELETE_OLD_IF_FALSE = False

# ============================================================
# SAFE MODE (prevents kernel crash)
# ============================================================
SAFE_MODE = True              #  keep True unless you really need in-loop GFLOPs
DO_GFLOPS_IN_LOOP = False     # keep False in SAFE_MODE

# ============================================================
# FIXED TRAIN SETTINGS
# ============================================================
MAIN_EPOCHS = 100
OPT_LR      = 1e-4

# ============================================================
# GUARDS
# ============================================================
MAX_PARAMS_M = 25.0
MAX_GFLOPS   = 200.0

# ============================================================
# SAVE ROOT (ACDC PATH)
# ============================================================
SAVE_ROOT = "/data/farhana/datasets/ACDC/ACDC_preprocessed"
CKPT_DIR  = os.path.join(SAVE_ROOT, "checkpoints")
LOG_DIR   = os.path.join(SAVE_ROOT, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.isdir(SAVE_ROOT):
    raise FileNotFoundError(f" SAVE_ROOT not found: {SAVE_ROOT}")

ENAS_STATE_PATH = os.path.join(LOG_DIR, "enas_pso_state.json")

# ============================================================
# JSON-safe RNG state pack/unpack
# ============================================================
def _pack_numpy_state(st):
    name, keys, pos, has_gauss, cached = st
    return {"name": str(name), "keys": keys.tolist(), "pos": int(pos),
            "has_gauss": int(has_gauss), "cached": float(cached)}

def _unpack_numpy_state(d):
    return (d["name"], np.array(d["keys"], dtype=np.uint32), int(d["pos"]),
            int(d["has_gauss"]), float(d["cached"]))

def _pack_python_state(st):
    ver, inner, gauss = copy.deepcopy(st)
    return {"ver": int(ver), "inner": list(inner), "gauss": gauss}

def _unpack_python_state(d):
    return (int(d["ver"]), tuple(d["inner"]), d["gauss"])

# ============================================================
#  SINGLE CANONICAL WeightedSum2 (TOP-LEVEL)
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

# ============================================================
#  FILTER SUPPORT (ADDED HERE ONLY)
# ============================================================
def _make_filters(base_filters):
    f = int(base_filters)
    if f < 8:
        f = 8
    elif f > 256:
        f = 256
    return (f // 8) * 8

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
# (Optional) True GFLOPs profiler (DO NOT use in SAFE_MODE loop)
# ============================================================
_GFLOPS_CACHE = {}

def _compute_gflops_silent(model, input_shape, batch_size=1):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    h, w, c = input_shape
    spec = tf.TensorSpec([batch_size, h, w, c], tf.float32)

    @tf.function
    def _forward(x):
        return model(x, training=False)

    concrete = _forward.get_concrete_function(spec)
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()

    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        with contextlib.redirect_stdout(io.StringIO()):
            prof = tf.compat.v1.profiler.profile(graph=g, run_meta=run_meta, cmd="op", options=opts)
        flops = prof.total_float_ops if prof is not None else 0
    return float(flops) / 1e9

def get_cached_gflops(model, input_shape, key):
    if key in _GFLOPS_CACHE:
        return float(_GFLOPS_CACHE[key])
    v = _compute_gflops_silent(model, input_shape, batch_size=1)
    _GFLOPS_CACHE[key] = float(v)
    return float(v)

# ============================================================
# SEARCH SPACE (ONLY your knobs)
# ============================================================
SEARCH_SPACE = {
    "base_filters":  list(range(8, 257, 8)),   
    "kernel_size":   [1, 3, 5],
    "activation":    ["relu", "elu", "softmax", "tanh"],
    "norm_type":     ["batch", "layer", "group"],
    "conv_type":     ["conv2d", "separableconv2d"],
    "pooling_type":  ["maxpool2d", "averagepool2d"],
    "fusion_method": ["add", "multiply"],

}
GENES = list(SEARCH_SPACE.keys())

def random_individual(input_shape=(256,256,1), num_classes=4):
    ind = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    ind["input_shape"] = tuple(input_shape)
    ind["num_classes"] = int(num_classes)
    ind["multilabel"]  = False
    ind["mutation_rate"] = random.random()
    ind["id"] = f"lid_{random.randint(0,10**12)}"
    ind["_obj"] = None
    ind["_rank"] = None
    ind["_crowd"] = None
    ind["_last_m"] = None
    ind["_best_m"] = None
    return ind

# ============================================================
# PSO Controller (mutation_rate in [0,1])
# PSO fitness will be rank/crowding based (no scalarize)
# ============================================================
class PSOController:
    def __init__(self, w=0.7, c1=1.6, c2=1.6, vmax=0.25, constriction=True):
        self.w, self.c1, self.c2 = float(w), float(c1), float(c2)
        self.vmax = float(vmax)
        self.constriction = bool(constriction)
        self.state = {}

    def _chi(self):
        phi = self.c1 + self.c2
        if phi <= 4:
            return 1.0
        return 2.0 / (abs(2 - phi - math.sqrt(phi**2 - 4*phi)))

    def ensure(self, lid, init_x):
        if lid not in self.state:
            self.state[lid] = {"x": float(init_x), "v": 0.0, "pbest": float(init_x), "pbest_fit": -np.inf}
        return self.state[lid]

    def suggest(self, lid, parent_mu1, parent_mu2, gbest_rate):
        x_i = 0.5 * (float(parent_mu1) + float(parent_mu2))
        s = self.ensure(lid, x_i)
        r1, r2 = random.random(), random.random()
        v_new = (
            self.w * s["v"]
            + self.c1 * r1 * (s["pbest"] - s["x"])
            + self.c2 * r2 * (float(gbest_rate) - s["x"])
        )
        if self.constriction:
            v_new *= self._chi()
        v_new = float(np.clip(v_new, -self.vmax, self.vmax))
        x_new = float(np.clip(s["x"] + v_new, 0.0, 1.0))
        s["_nx"], s["_nv"] = x_new, v_new
        return x_new

    def update(self, lid, achieved_fit):
        if lid not in self.state:
            return
        s = self.state[lid]
        if "_nx" in s and "_nv" in s:
            s["x"], s["v"] = s.pop("_nx"), s.pop("_nv")
        if float(achieved_fit) > s["pbest_fit"]:
            s["pbest_fit"] = float(achieved_fit)
            s["pbest"] = float(s["x"])

# ============================================================
# LLM (delta_mu, targets, intensity)
# If OpenAI SDK/API key not available, it falls back to neutral outputs.
# ============================================================
LLM_ENABLED = True
LLM_MODEL   = "gpt-5"
DELTA_CLIP  = 0.10
LLM_CALL_EVERY_N_CHILDREN = 1

try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None
    LLM_ENABLED = False
    print(" OpenAI SDK not available or API key missing. LLM guidance fallback to neutral.")

class LLMMutationAdvisor:
    def __init__(self, delta_clip=0.10, temperature=0.0, cache=True):
        self.delta_clip = float(delta_clip)
        self.temperature = float(temperature)
        self.cache = bool(cache)
        self._cache = {}
        self._calls = 0

    def _schema(self):
        return {
            "type": "object",
            "properties": {
                "delta_mu": {"type": "number", "minimum": -self.delta_clip, "maximum": self.delta_clip},
                "targets": {
                    "type": "array",
                    "items": {"type": "string", "enum": GENES},
                    "minItems": 0,
                    "uniqueItems": True
                },
                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"}
            },
            "required": ["delta_mu", "targets", "intensity"],
            "additionalProperties": False
        }

    def propose(self, context: dict):
        self._calls += 1
        if LLM_CALL_EVERY_N_CHILDREN > 1:
            if (self._calls % int(LLM_CALL_EVERY_N_CHILDREN)) != 0:
                return {"delta_mu": 0.0, "targets": [], "intensity": 0.0}

        if (not LLM_ENABLED) or (_openai_client is None):
            return {"delta_mu": 0.0, "targets": [], "intensity": 0.0}

        key = None
        if self.cache:
            key = json.dumps(context, sort_keys=True)
            if key in self._cache:
                return dict(self._cache[key])

        sys_msg = (
            "You are a mutation-operator advisor for multi-objective evolutionary NAS.\n"
            "Return JSON matching the schema.\n"
            f"delta_mu in [-{self.delta_clip}, +{self.delta_clip}] (small)\n"
            "targets: subset of gene names to mutate\n"
            "intensity in [0,1]: focus strength on targets\n"
            "Guidance:\n"
            "- If rank is good but crowding is low: increase intensity & diversify targets.\n"
            "- If rank is worse: delta_mu positive (explore) & intensity moderate.\n"
            "- Prefer small changes.\n"
        )
        user_msg = f"Context:\n{json.dumps(context)}"

        try:
            resp = _openai_client.responses.create(
                model=LLM_MODEL,
                input=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                text={"format": {"type": "json_schema", "json_schema": self._schema()}},
                temperature=self.temperature,
            )
            data = json.loads(resp.output_text)
            delta_mu = float(np.clip(float(data["delta_mu"]), -self.delta_clip, self.delta_clip))
            targets = [t for t in list(data.get("targets", [])) if t in GENES]
            intensity = float(np.clip(float(data.get("intensity", 0.0)), 0.0, 1.0))
            out = {"delta_mu": delta_mu, "targets": targets, "intensity": intensity}
        except Exception:
            out = {"delta_mu": 0.0, "targets": [], "intensity": 0.0}

        if self.cache and key is not None:
            self._cache[key] = dict(out)
        return out

# ============================================================
# Resume state IO
# ============================================================
def _serialize_ind(ind):
    keep = ["id","mutation_rate","input_shape","num_classes","multilabel"] + GENES
    out = {k: ind[k] for k in keep}
    out["_obj"] = ind.get("_obj", None)
    out["_rank"] = ind.get("_rank", None)
    out["_crowd"] = ind.get("_crowd", None)
    out["_best_m"] = ind.get("_best_m", None)
    out["_last_m"] = ind.get("_last_m", None)
    return out

def _deserialize_ind(d):
    ind = dict(d)
    ind.setdefault("_obj", None)
    ind.setdefault("_rank", None)
    ind.setdefault("_crowd", None)
    ind.setdefault("_best_m", None)
    ind.setdefault("_last_m", None)
    return ind

def save_enas_state(gen_idx, ind_idx, pop, pso_state, gbest_mu, seed):
    rng_state = {
        "python_random": _pack_python_state(random.getstate()),
        "numpy_random":  _pack_numpy_state(np.random.get_state()),
        "tf_seed": int(seed),
    }
    payload = {
        "gen_idx": int(gen_idx),
        "ind_idx": int(ind_idx),
        "gbest_mu": float(gbest_mu),
        "population": [_serialize_ind(p) for p in pop],
        "pso_state": pso_state,
        "rng_state": rng_state,
    }
    tmp = ENAS_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, ENAS_STATE_PATH)

def load_enas_state():
    if not os.path.exists(ENAS_STATE_PATH):
        return None
    with open(ENAS_STATE_PATH, "r") as f:
        return json.load(f)

def maybe_reset_state():
    if RESUME_ENAS:
        return
    if DELETE_OLD_IF_FALSE and os.path.exists(ENAS_STATE_PATH):
        os.remove(ENAS_STATE_PATH)
        print(" Deleted ENAS state:", ENAS_STATE_PATH)

# ============================================================
# NSGA-II helpers (minimize vector)   resume-safe
# ============================================================
def dominates(a, b):
    va, vb = a.get("_obj", None), b.get("_obj", None)
    if va is None or vb is None:
        return False
    return all(x <= y for x, y in zip(va, vb)) and any(x < y for x, y in zip(va, vb))

def fast_nondominated_sort(pop):
    pop = [p for p in pop if p.get("_obj", None) is not None]
    if not pop:
        return [[]]

    fronts = []
    S = {id(p): [] for p in pop}
    n = {id(p): 0 for p in pop}
    F0 = []
    for p in pop:
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                S[id(p)].append(q)
            elif dominates(q, p):
                n[id(p)] += 1
        if n[id(p)] == 0:
            p["_rank"] = 0
            F0.append(p)
    fronts.append(F0)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[id(p)]:
                n[id(q)] -= 1
                if n[id(q)] == 0:
                    q["_rank"] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts[:-1]

def crowding_distance(front):
    if not front:
        return
    m = len(front[0]["_obj"])
    for p in front:
        p["_crowd"] = 0.0
    for j in range(m):
        front.sort(key=lambda x: x["_obj"][j])
        front[0]["_crowd"] = front[-1]["_crowd"] = float("inf")
        vmin = front[0]["_obj"][j]
        vmax = front[-1]["_obj"][j]
        if vmax - vmin < 1e-12:
            continue
        for i in range(1, len(front) - 1):
            front[i]["_crowd"] += (front[i+1]["_obj"][j] - front[i-1]["_obj"][j]) / (vmax - vmin)

def nsga2_select(pop, n_keep):
    fronts = fast_nondominated_sort(pop)
    selected = []
    for F in fronts:
        crowding_distance(F)
        if len(selected) + len(F) <= n_keep:
            selected.extend(F)
        else:
            F.sort(key=lambda x: (x["_rank"], -x["_crowd"]))
            selected.extend(F[: (n_keep - len(selected))])
            break
    return selected

def tournament(pop, k=2):
    cand = random.sample(pop, k)
    cand.sort(key=lambda x: (x["_rank"], -x["_crowd"]))
    return cand[0]

# ============================================================
# Pareto-consistent PSO fitness (rank/crowding only)
# ============================================================
def compute_rank_crowd_for_done(done):
    if not done:
        return {}

    fronts = fast_nondominated_sort(done)
    rc = {}

    for front in fronts:
        crowding_distance(front)

        finite_vals = [p["_crowd"] for p in front if not np.isinf(p["_crowd"])]
        cmin = min(finite_vals) if finite_vals else 0.0
        cmax = max(finite_vals) if finite_vals else 1.0

        for p in front:
            crowd = p["_crowd"]
            if np.isinf(crowd):
                cn = 1.0
            elif cmax - cmin < 1e-12:
                cn = 0.0
            else:
                cn = (crowd - cmin) / (cmax - cmin)

            rc[p["id"]] = (p["_rank"], crowd, cn)

    return rc


def pso_fitness_from_rank_crowd(rank, crowd_norm, crowd_weight=0.05):
    return float(-float(rank) + float(crowd_weight) * float(crowd_norm))

# ============================================================
# Print config BEFORE training
# ============================================================
def print_arch(ind, gen, idx):
    print(f"\n--- ARCH gen{gen} ind{idx} ---")
    for k in GENES:
        print(f"{k}: {ind[k]}")
    print(f"mutation_rate: {float(ind.get('mutation_rate', 0.0)):.4f}")

# ============================================================
# Evaluate individual (guards + training)   SAFE MODE stable
# Objective vector minimized: (-dice, -iou, hd95, gflops_proxy, params_m)
# ============================================================
def evaluate_individual(ind, gen, idx, train_seq, val_seq):
    arch_id   = f"enas_g{gen}_i{idx}_{ind['id']}"
    arch_ckpt = os.path.join(CKPT_DIR, f"{arch_id}.keras")
    arch_json = os.path.join(LOG_DIR,  f"{arch_id}.json")

    print_arch(ind, gen, idx)

    model = model_arch(
        input_shape=ind["input_shape"],
        num_classes=ind["num_classes"],
        base_filters=int(ind["base_filters"]),
        kernel_size=int(ind["kernel_size"]),
        activation=str(ind["activation"]),
        norm_type=str(ind["norm_type"]),
        conv_type=str(ind["conv_type"]),
        pooling_type=str(ind["pooling_type"]),
        fusion_method=str(ind["fusion_method"]),
        dropout_deep=0.2,
        multilabel=bool(ind.get("multilabel", False)),
    )

    params_m = model.count_params() / 1e6
    print(f"paramsM: {params_m:.3f}")

    # ---- cheap GFLOPs proxy during search (stable, no profiler) ----
    gflops_proxy = float(params_m)

    # Params guard (always)
    if params_m > MAX_PARAMS_M:
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        dice, iou, hd95 = 0.0, 0.0, 95.0
        obj = (-dice, -iou, hd95, gflops_proxy, params_m)
        out = {"arch_id": arch_id, "gen": int(gen), "idx": int(idx),
               "skipped": True, "reason": f"PARAM_GUARD paramsM={params_m:.3f}",
               "dice": dice, "iou": iou, "hd95": hd95,
               "gflops": None, "gflops_proxy": float(gflops_proxy),
               "params_m": float(params_m),
               "mutation_rate": float(ind.get("mutation_rate", 0.1)),
               "config": {k: ind[k] for k in (GENES + ["mutation_rate"])}}
        with open(arch_json, "w") as f:
            json.dump(out, f, indent=2)
        return obj, out

    # OPTIONAL true GFLOPs guard (NOT in SAFE_MODE)
    if (not SAFE_MODE) and DO_GFLOPS_IN_LOOP:
        gkey = tuple(ind[k] for k in GENES)
        gflops_true = get_cached_gflops(model, ind["input_shape"], gkey)
        print(f"gflops(TRUE): {gflops_true:.3f}")
        if gflops_true > MAX_GFLOPS:
            tf.keras.backend.clear_session()
            del model
            gc.collect()
            dice, iou, hd95 = 0.0, 0.0, 95.0
            obj = (-dice, -iou, hd95, float(gflops_true), params_m)
            out = {"arch_id": arch_id, "gen": int(gen), "idx": int(idx),
                   "skipped": True, "reason": f"GFLOPS_GUARD gflops={gflops_true:.3f}",
                   "dice": float(dice), "iou": float(iou), "hd95": float(hd95),
                   "gflops": float(gflops_true), "gflops_proxy": float(gflops_proxy),
                   "params_m": float(params_m),
                   "mutation_rate": float(ind.get("mutation_rate", 0.1)),
                   "config": {k: ind[k] for k in (GENES + ["mutation_rate"])}}
            with open(arch_json, "w") as f:
                json.dump(out, f, indent=2)
            return obj, out

    opt = tf.keras.optimizers.Adam(OPT_LR)
    model.compile(
        optimizer=opt,
        loss=ce_tversky_multiclass(lam=0.7, alpha=0.3, beta=0.7, exclude_bg=True),
        metrics=[
            MeanDiceForeground(num_classes=ind["num_classes"], name="mean_dice"),
            MeanIoUForeground(num_classes=ind["num_classes"], name="mean_iou"),
            MeanHD95Foreground(num_classes=ind["num_classes"], name="mean_hd95"),
        ],
        jit_compile=False,
    )

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        arch_ckpt, monitor="val_mean_dice", mode="max", save_best_only=True, verbose=0
    )
    early_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mean_dice",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1
    )

    t0 = time.time()
    hist = None
    try:
        hist = model.fit(train_seq, validation_data=val_seq, epochs=MAIN_EPOCHS, verbose=1, callbacks=[ckpt_cb, early_cb])
    except tf.errors.ResourceExhaustedError:
        tf.keras.backend.clear_session()
        try: del model
        except Exception: pass
        gc.collect()
        dice, iou, hd95 = 0.0, 0.0, 95.0
        obj = (-dice, -iou, hd95, gflops_proxy, params_m)
        out = {"arch_id": arch_id, "gen": int(gen), "idx": int(idx),
               "skipped": True, "reason": "OOM_DURING_FIT",
               "dice": dice, "iou": iou, "hd95": hd95,
               "gflops": None, "gflops_proxy": float(gflops_proxy),
               "params_m": float(params_m),
               "mutation_rate": float(ind.get("mutation_rate", 0.1)),
               "config": {k: ind[k] for k in (GENES + ["mutation_rate"])}}
        with open(arch_json, "w") as f:
            json.dump(out, f, indent=2)
        return obj, out
    tsec = time.time() - t0

    # Load best (optional)
    custom_map = {"WeightedSum2": WeightedSum2, "Custom>WeightedSum2": WeightedSum2, "Custom.WeightedSum2": WeightedSum2}
    if os.path.exists(arch_ckpt):
        try:
            model = tf.keras.models.load_model(arch_ckpt, custom_objects=custom_map, compile=False, safe_mode=False)
            model.compile(
                optimizer=opt,
                loss=ce_tversky_multiclass(lam=0.7, alpha=0.3, beta=0.7, exclude_bg=True),
                metrics=[
                    MeanDiceForeground(num_classes=ind["num_classes"], name="mean_dice"),
                    MeanIoUForeground(num_classes=ind["num_classes"], name="mean_iou"),
                    MeanHD95Foreground(num_classes=ind["num_classes"], name="mean_hd95"),
                ],
                jit_compile=False,
            )
        except Exception:
            pass

    # Evaluate
    res = model.evaluate(val_seq, verbose=0)
    m = {n: float(v) for n, v in zip(model.metrics_names, res)}
    dice = m.get("mean_dice", float(hist.history.get("val_mean_dice", [0.0])[-1]) if hist is not None else 0.0)
    iou  = m.get("mean_iou",  float(hist.history.get("val_mean_iou",  [0.0])[-1]) if hist is not None else 0.0)
    hd95 = m.get("mean_hd95", float(hist.history.get("val_mean_hd95", [95.0])[-1]) if hist is not None else 95.0)

    # Objective vector (minimize)
    obj = (-float(dice), -float(iou), float(hd95), float(gflops_proxy), float(params_m))

    out = {
        "arch_id": arch_id, "gen": int(gen), "idx": int(idx),
        "time_sec": float(tsec),
        "dice": float(dice), "iou": float(iou), "hd95": float(hd95),
        "gflops": None, "gflops_proxy": float(gflops_proxy),
        "params_m": float(params_m),
        "mutation_rate": float(ind.get("mutation_rate", 0.1)),
        "config": {k: ind[k] for k in (GENES + ["mutation_rate"])},
    }
    with open(arch_json, "w") as f:
        json.dump(out, f, indent=2)

    # Aggressive cleanup (prevents kernel death over time)
    try: del res
    except Exception: pass
    try: del hist
    except Exception: pass
    try: del model
    except Exception: pass
    tf.keras.backend.clear_session()
    gc.collect()

    return obj, out

# ============================================================
# Crossover + mutate (PSO + LLM: delta_mu, targets, intensity)
# ============================================================
def crossover_and_mutate(p1, p2, pso, llm, child_id, gbest_mu, gen=None, alpha_pso=0.65):
    child = {"id": child_id}
    for k in GENES:
        child[k] = random.choice([p1[k], p2[k]])

    mu1 = float(p1.get("mutation_rate", 0.1))
    mu2 = float(p2.get("mutation_rate", 0.1))
    mu_avg = 0.5 * (mu1 + mu2)

    mu_pso = float(pso.suggest(child_id, mu1, mu2, gbest_rate=float(gbest_mu)))

    best_parent = p1 if int(p1.get("_rank", 10**9)) < int(p2.get("_rank", 10**9)) else p2
    ctx = {
        "gen": int(gen) if gen is not None else None,
        "mu_avg": float(mu_avg),
        "gbest_mu": float(gbest_mu),
        "parent_rank": int(best_parent.get("_rank", 10**9)) if best_parent.get("_rank", None) is not None else None,
        "parent_crowd": float(best_parent.get("_crowd", 0.0)) if best_parent.get("_crowd", None) is not None else None,
        "parent_last": best_parent.get("_last_m", None),
        "genes": GENES,
        "search_space_sizes": {k: len(SEARCH_SPACE[k]) for k in GENES},
    }
    prop = llm.propose(ctx)
    delta_mu = float(np.clip(float(prop.get("delta_mu", 0.0)), -DELTA_CLIP, DELTA_CLIP))
    targets = [t for t in list(prop.get("targets", [])) if t in GENES]
    intensity = float(np.clip(float(prop.get("intensity", 0.0)), 0.0, 1.0))

    mu_llm = float(np.clip(mu_avg + delta_mu, 0.0, 1.0))
    mu_child = float(np.clip(alpha_pso * mu_pso + (1.0 - alpha_pso) * mu_llm, 0.0, 1.0))
    child["mutation_rate"] = mu_child

    bg_scale = 0.25
    for k in GENES:
        if k in targets:
            p_mut = mu_child * intensity
        else:
            p_mut = mu_child * (1.0 - intensity) * bg_scale
        if random.random() < p_mut:
            child[k] = random.choice(SEARCH_SPACE[k])

    child["input_shape"] = p1["input_shape"]
    child["num_classes"] = p1["num_classes"]
    child["multilabel"]  = p1.get("multilabel", False)

    child["_obj"] = None
    child["_rank"] = None
    child["_crowd"] = None
    child["_best_m"] = None
    child["_last_m"] = None
    return child

# ============================================================
# Main runner with resume cursor (resume-safe)
# ============================================================
def run_pso_enas_resume(train_seq, val_seq, pop_size=50, generations=10, seed=42):
    maybe_reset_state()
    state = load_enas_state() if RESUME_ENAS else None

    if state is not None:
        print(" Resuming ENAS from:", ENAS_STATE_PATH)
        gen0 = int(state["gen_idx"])
        ind0 = int(state["ind_idx"])
        gbest_mu = float(state.get("gbest_mu", 0.5))
        pop = [_deserialize_ind(p) for p in state["population"]]
        pso = PSOController()
        pso.state = state.get("pso_state", {})
        llm = LLMMutationAdvisor(delta_clip=DELTA_CLIP, temperature=0.0, cache=True)

        rs = state.get("rng_state", None)
        if rs is not None:
            try: random.setstate(_unpack_python_state(rs["python_random"]))
            except Exception: pass
            try: np.random.set_state(_unpack_numpy_state(rs["numpy_random"]))
            except Exception: pass
            tf.random.set_seed(int(rs.get("tf_seed", seed)))
    else:
        print(" New ENAS run")
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
        gen0, ind0 = 0, 0
        pop = [random_individual(input_shape=(256, 256, 1), num_classes=4) for _ in range(pop_size)]
        gbest_mu = float(np.median([p["mutation_rate"] for p in pop]))
        pso = PSOController()
        llm = LLMMutationAdvisor(delta_clip=DELTA_CLIP, temperature=0.0, cache=True)

    for gen in range(gen0, generations):
        print(f"\n================== GEN {gen}/{generations-1} ==================")
        start_i = ind0 if gen == gen0 else 0

        # ---- evaluate pop ----
        for i in range(start_i, pop_size):
            ind = pop[i]
            obj, info = evaluate_individual(ind, gen, i, train_seq, val_seq)
            ind["_obj"] = obj

            ind["_last_m"] = {
                "dice": info.get("dice", 0.0),
                "iou": info.get("iou", 0.0),
                "hd95": info.get("hd95", 95.0),
                "gflops_proxy": info.get("gflops_proxy", float("nan")),
                "params_m": info.get("params_m", 1e9)
            }

            done = [p for p in pop if p.get("_obj", None) is not None]
            rc = compute_rank_crowd_for_done(done)
            r, c, cn = rc.get(ind["id"], (10**9, 0.0, 0.0))
            ind["_rank"], ind["_crowd"] = int(r), float(c)

            fit = pso_fitness_from_rank_crowd(rank=r, crowd_norm=cn, crowd_weight=0.05)
            pso.ensure(ind["id"], float(ind.get("mutation_rate", 0.1)))
            pso.update(ind["id"], fit)

            done.sort(key=lambda x: (int(x.get("_rank", 10**9)),
                                     -float(x.get("_crowd", 0.0) if x.get("_crowd", None) is not None else 0.0)))
            if done:
                gbest_mu = float(done[0]["mutation_rate"])

            print(f"[{i+1}/{pop_size}] dice={info.get('dice',0.0):.4f} iou={info.get('iou',0.0):.4f} hd95={info.get('hd95',95.0):.3f} "
                  f"gflops_proxy={info.get('gflops_proxy',float('nan')):.3f} paramsM={info.get('params_m',1e9):.3f} "
                  f"mu={ind['mutation_rate']:.3f} rank={int(ind.get('_rank',999))} crowd={float(ind.get('_crowd',0.0)):.3f} "
                  f"{'(SKIP)' if info.get('skipped', False) else ''}")

            save_enas_state(gen_idx=gen, ind_idx=i+1, pop=pop, pso_state=pso.state, gbest_mu=gbest_mu, seed=seed)

        # ---- end-of-gen Pareto summary (resume-safe) ----
        done = [p for p in pop if p.get("_obj", None) is not None]
        fronts = fast_nondominated_sort(done)
        for F in fronts:
            crowding_distance(F)
        pareto = fronts[0] if fronts else []

        print(f"\nPareto Front size: {len(pareto)}")
        for j, p in enumerate(pareto[: min(5, len(pareto))]):
            dice = -p["_obj"][0]; iou = -p["_obj"][1]; hd95 = p["_obj"][2]; gfpx = p["_obj"][3]; pm = p["_obj"][4]
            print(f"  #{j+1}: dice={dice:.4f} iou={iou:.4f} hd95={hd95:.3f} gflops_proxy={gfpx:.3f} paramsM={pm:.3f} mu={p['mutation_rate']:.3f}")

        # ---- offspring ----
        parents = nsga2_select(done, pop_size) if len(done) > 0 else []
        if not parents:
            parents = [random_individual(input_shape=(256,256,1), num_classes=4) for _ in range(pop_size)]
            for p in parents:
                p["_rank"], p["_crowd"] = 999, 0.0

        offspring = []
        for k in range(pop_size):
            a = tournament(parents, k=2) if len(parents) >= 2 else parents[0]
            b = tournament(parents, k=2) if len(parents) >= 2 else parents[0]
            child_id = f"lid_g{gen+1}_c{k}_{random.randint(0,10**9)}"
            child = crossover_and_mutate(a, b, pso, llm, child_id, gbest_mu, gen=gen, alpha_pso=0.65)
            offspring.append(child)

        pop = offspring
        save_enas_state(gen_idx=gen+1, ind_idx=0, pop=pop, pso_state=pso.state, gbest_mu=gbest_mu, seed=seed)

        # extra cleanup per generation
        tf.keras.backend.clear_session()
        gc.collect()

    return pareto

# ============================================================
# RUN
# ============================================================
assert "train_seq" in globals() and "val_seq" in globals(), "train_seq / val_seq not found. Build data loaders first."
print("len(train_seq)=", len(train_seq), "| len(val_seq)=", len(val_seq))
final_pareto = run_pso_enas_resume(train_seq, val_seq, pop_size=50, generations=10, seed=42)
print("\n DONE. final_pareto length =", len(final_pareto))

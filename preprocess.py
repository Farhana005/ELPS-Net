import os
import numpy as np
import tensorflow as tf
import h5py
from collections import OrderedDict
from tensorflow.keras.utils import Sequence
from tensorflow.keras import mixed_precision

# ---- HD95 needs SciPy ----
from scipy.ndimage import distance_transform_edt

# -------------------------
# GLOBAL AMP + NO XLA
# -------------------------
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(False)
print("Policy:", mixed_precision.global_policy(), "| XLA:", tf.config.optimizer.get_jit())
print("GPUs:", tf.config.list_physical_devices("GPU"))

# -------------------------
# CONFIG


TRAIN_H5_DIR = "/data2/farhana/datasets/ACDC/ACDC_preprocessed/ACDC_training_volumes"
VAL_H5_DIR   = "/data2/farhana/datasets/ACDC/ACDC_preprocessed/ACDC_testing_volumes"

TARGET_SIZE  = (256, 256)
NUM_CLASSES  = 4     # 0=bg, 1=RV, 2=MYO, 3=LV
SEED = 42

# sampling per case (Option B+)
TRAIN_CASES_PER_EPOCH = 60
TRAIN_K_RV  = 3
TRAIN_K_MYO = 3
TRAIN_K_LV  = 3
TRAIN_K_NEG = 4
TRAIN_BATCH = 8

VAL_CASES_PER_EPOCH = 20
VAL_K_RV  = 3
VAL_K_MYO = 3
VAL_K_LV  = 3
VAL_K_NEG = 0
VAL_BATCH = 8
VAL_FREEZE_SLICES = True

CACHE_CASES = 8
NORMALIZE = "minmax"   # "minmax" | "zscore_nonzero" | None

# -------------------------
# AUGMENTATION (TRAIN ONLY)
# -------------------------
DO_AUGMENT_TRAIN = True
AUG_FLIP_LR_PROB = 0.5
AUG_BRIGHT_PROB  = 0.5
AUG_NOISE_PROB   = 0.3
BRIGHT_DELTA     = 0.08
CONTRAST_LOW     = 0.90
CONTRAST_HIGH    = 1.10
NOISE_STD        = 0.02

# -------------------------
# LABEL MAP (canonical)
# -------------------------
LABEL_MAP = {0:0, 1:1, 2:2, 3:3}

def remap_labels_np(lab_dhw, label_map):
    lab = lab_dhw.astype(np.int32)
    out = np.zeros_like(lab, dtype=np.int32)
    for k, v in label_map.items():
        out[lab == int(k)] = int(v)
    return out

# -------------------------
# Helpers
# -------------------------
def _is_valid_h5(p):
    return (p is not None) and os.path.exists(p) and (os.path.getsize(p) > 0) and p.endswith(".h5")

def _list_h5_files(root_dir):
    try:
        return sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".h5")])
    except Exception:
        return []

def _safe_read_h5(path, image_key="image", label_key="label"):
    if not _is_valid_h5(path):
        return None, None
    try:
        with h5py.File(path, "r") as f:
            if image_key not in f:
                return None, None
            img = f[image_key][:]
            lab = f[label_key][:] if (label_key in f) else None
        return img, lab
    except Exception:
        return None, None

def _to_dhw(img):
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[None, ...]  # (1,H,W)
    if img.ndim in (3,4):
        return img
    return None

def _to_label_dhw(lab):
    if lab is None:
        return None
    lab = np.asarray(lab)
    if lab.ndim == 2:
        lab = lab[None, ...]
    if lab.ndim == 3:
        return lab
    return None

def _minmax(vol):
    vol = vol.astype(np.float32)
    vmin = np.min(vol)
    vmax = np.max(vol)
    if vmax - vmin < 1e-6:
        return vol
    return (vol - vmin) / (vmax - vmin)

def _zscore_nonzero(vol):
    vol = vol.astype(np.float32)
    m = vol != 0
    if np.any(m):
        mu = vol[m].mean()
        sd = vol[m].std() + 1e-6
        vol[m] = (vol[m] - mu) / sd
    return vol

def _resize_img_tf(img_hw_c, target_size=(256, 256)):
    x = tf.convert_to_tensor(img_hw_c, dtype=tf.float32)
    return tf.image.resize(x, target_size, method="bilinear")

def _resize_mask_tf(mask_hw, target_size=(256, 256)):
    m = tf.convert_to_tensor(mask_hw, dtype=tf.float32)
    m = tf.image.resize(m[..., None], target_size, method="nearest")
    m = tf.squeeze(m, axis=-1)
    return tf.cast(m, tf.int32)

def _one_hot(mask_hw_int, num_classes):
    return tf.one_hot(mask_hw_int, depth=num_classes, dtype=tf.float32)

def _augment_pair_safe(x, y, rng):
    if rng.random() < AUG_FLIP_LR_PROB:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    if rng.random() < AUG_BRIGHT_PROB:
        x = tf.image.random_brightness(x, max_delta=BRIGHT_DELTA)
        x = tf.image.random_contrast(x, lower=CONTRAST_LOW, upper=CONTRAST_HIGH)

    if rng.random() < AUG_NOISE_PROB:
        x = x + tf.random.normal(tf.shape(x), stddev=NOISE_STD)

    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y

# -------------------------
# Build case list + slice pools
# -------------------------
def build_acdc_case_list(h5_dir, require_label=True, image_key="image", label_key="label"):
    paths = _list_h5_files(h5_dir)
    cases = []
    for p in paths:
        img, lab = _safe_read_h5(p, image_key=image_key, label_key=label_key)
        if img is None:
            continue
        if require_label and lab is None:
            continue

        img = _to_dhw(img)
        lab = _to_label_dhw(lab) if lab is not None else None
        if img is None:
            continue
        if require_label and lab is None:
            continue

        lab = remap_labels_np(lab, LABEL_MAP)

        if img.ndim == 3:
            D = img.shape[0]
        else:
            D = img.shape[0]

        # sanity
        if lab.shape[0] != D:
            continue

        # compute per-slice pools in original D,H,W layout
        # lab: (D,H,W)
        mx = lab.max(axis=(1, 2))
        pos_any = np.where(mx > 0)[0].tolist()
        neg_z   = np.where(mx == 0)[0].tolist()

        pos_rv  = np.where((lab == 1).any(axis=(1, 2)))[0].tolist()
        pos_myo = np.where((lab == 2).any(axis=(1, 2)))[0].tolist()
        pos_lv  = np.where((lab == 3).any(axis=(1, 2)))[0].tolist()

        cases.append(
            dict(
                name=os.path.splitext(os.path.basename(p))[0],
                path=p,
                pos_any=pos_any,
                pos_rv=pos_rv,
                pos_myo=pos_myo,
                pos_lv=pos_lv,
                neg_z=neg_z,
            )
        )
    return cases

# -------------------------
# Sequence (LRU cache + RV-aware sampling) — Keras 3 SAFE
#   - stable __len__ (fixed steps/epoch)
#   - __getitem__ samples WITH replacement => never "runs out of data"
# -------------------------
class ACDCCaseWiseSliceSequence(Sequence):
    def __init__(
        self,
        cases,
        case_names,
        target_size=(256, 256),
        batch_size=8,
        cases_per_epoch=60,
        k_rv=3, k_myo=3, k_lv=3, k_neg=3,
        num_classes=4,
        normalize="minmax",
        shuffle_cases=True,
        seed=42,
        freeze_slices=False,
        cache_cases=8,
        image_key="image",
        label_key="label",
        do_augment=False,
        **kwargs,                 
    ):
        super().__init__(**kwargs)  
        self.cases_all = list(cases)
        self.target_size = tuple(target_size)
        self.batch_size = int(batch_size)
        self.cases_per_epoch = int(cases_per_epoch) if cases_per_epoch is not None else None

        self.k_rv, self.k_myo, self.k_lv, self.k_neg = int(k_rv), int(k_myo), int(k_lv), int(k_neg)
        self.num_classes = int(num_classes)

        self.normalize = normalize
        self.shuffle_cases = bool(shuffle_cases)
        self.freeze_slices = bool(freeze_slices)

        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        self.image_key = image_key
        self.label_key = label_key
        self.do_augment = bool(do_augment)

        self.name_to_case = {c["name"]: c for c in self.cases_all}
        self.case_names = [n for n in case_names if n in self.name_to_case]
        if len(self.case_names) == 0:
            raise ValueError("No valid case names provided to the Sequence.")

        self.cache_cases = int(cache_cases)
        self._vol_cache = OrderedDict()  # name -> (img(H,W,D,C) float32, lab(H,W,D) int32)

        self.slice_pool = []
        self._pool_built_once = False
        self.epoch = 0

        self.on_epoch_end()

    # ----- cache -----
    def _cache_get(self, name):
        v = self._vol_cache.get(name, None)
        if v is not None:
            self._vol_cache.move_to_end(name)
        return v

    def _cache_put(self, name, value):
        self._vol_cache[name] = value
        self._vol_cache.move_to_end(name)
        while len(self._vol_cache) > self.cache_cases:
            self._vol_cache.popitem(last=False)

    def _load_case_volumes(self, name):
        cached = self._cache_get(name)
        if cached is not None:
            return cached

        case = self.name_to_case[name]
        img_np, lab_np = _safe_read_h5(case["path"], image_key=self.image_key, label_key=self.label_key)
        if img_np is None or lab_np is None:
            return None

        img_np = _to_dhw(img_np)
        lab_np = _to_label_dhw(lab_np)
        if img_np is None or lab_np is None:
            return None

        lab_np = remap_labels_np(lab_np, LABEL_MAP)

        # Expect (D,H,W) or (D,H,W,C)
        if img_np.ndim == 3:
            D, H, W = img_np.shape
            img = np.transpose(img_np, (1, 2, 0))[..., None].astype(np.float32)  # (H,W,D,1)
        else:
            D, H, W, C = img_np.shape
            img = np.transpose(img_np, (1, 2, 0, 3)).astype(np.float32)          # (H,W,D,C)

        if lab_np.shape[0] != D:
            return None
        lab = np.transpose(lab_np, (1, 2, 0)).astype(np.int32)  # (H,W,D)

        if self.normalize == "minmax":
            img = _minmax(img)
        elif self.normalize == "zscore_nonzero":
            for c in range(img.shape[-1]):
                img[..., c] = _zscore_nonzero(img[..., c])

        out = (img, lab)
        self._cache_put(name, out)
        return out

    # ----- build pool once per epoch (or once total if freeze_slices=True) -----
    def on_epoch_end(self):
        if self.freeze_slices and self._pool_built_once:
            self.epoch += 1
            return

        self.epoch += 1

        names = list(self.case_names)

        # choose cases for this epoch
        if self.cases_per_epoch is None or self.cases_per_epoch >= len(names):
            chosen = names
        else:
            chosen = list(self.rng.choice(names, size=self.cases_per_epoch, replace=False))

        if self.shuffle_cases and not self.freeze_slices:
            self.rng.shuffle(chosen)

        pool = []
        for name in chosen:
            case = self.name_to_case[name]
            rv, myo, lv = case.get("pos_rv", []), case.get("pos_myo", []), case.get("pos_lv", [])
            any_fg = case.get("pos_any", [])
            neg = case.get("neg_z", [])

            def pick(from_list, k):
                if k <= 0:
                    return []
                if len(from_list) > 0:
                    return list(self.rng.choice(from_list, size=k, replace=(len(from_list) < k)))
                if len(any_fg) > 0:
                    return list(self.rng.choice(any_fg, size=k, replace=(len(any_fg) < k)))
                if len(neg) > 0:
                    return list(self.rng.choice(neg, size=k, replace=(len(neg) < k)))
                return []

            for z in pick(rv,  self.k_rv):  pool.append((name, int(z)))
            for z in pick(myo, self.k_myo): pool.append((name, int(z)))
            for z in pick(lv,  self.k_lv):  pool.append((name, int(z)))

            if self.k_neg > 0 and len(neg) > 0:
                neg_pick = self.rng.choice(neg, size=self.k_neg, replace=(len(neg) < self.k_neg))
                for z in neg_pick:
                    pool.append((name, int(z)))

        if not self.freeze_slices:
            self.rng.shuffle(pool)

        # if somehow too small, replicate (safety)
        if len(pool) == 0:
            # keep non-empty to avoid crashes; will return zeros
            self.slice_pool = []
        else:
            min_needed = max(self.batch_size, 2 * self.batch_size)  # a little buffer
            if len(pool) < min_needed:
                times = int(np.ceil(min_needed / len(pool)))
                pool = (pool * times)[:min_needed]
            self.slice_pool = pool

        self._pool_built_once = True

    # ----- FIXED steps/epoch: stable -----
    def __len__(self):
        # target a stable number of steps based on expected slices per epoch
        if self.cases_per_epoch is None:
            c = len(self.case_names)
        else:
            c = self.cases_per_epoch
        slices_per_case = self.k_rv + self.k_myo + self.k_lv + self.k_neg
        total = int(c * slices_per_case)
        return int(np.ceil(total / self.batch_size))

    # ----- NEVER exhaust: sample with replacement -----
    def __getitem__(self, idx):
        if len(self.slice_pool) == 0:
            x = tf.zeros((self.batch_size, self.target_size[0], self.target_size[1], 1), tf.float16)
            y = tf.zeros((self.batch_size, self.target_size[0], self.target_size[1], self.num_classes), tf.float32)
            return x, y

        # deterministic per (epoch, idx) but still "random"
        local_seed = (self.base_seed * 1000003 + self.epoch * 9176 + idx * 101) & 0xFFFFFFFF
        rrng = np.random.default_rng(local_seed)

        xb, yb = [], []
        pool_len = len(self.slice_pool)

        attempts = 0
        max_attempts = max(200, self.batch_size * 30)

        while len(xb) < self.batch_size and attempts < max_attempts:
            attempts += 1

            name, z = self.slice_pool[int(rrng.integers(0, pool_len))]

            loaded = self._load_case_volumes(name)
            if loaded is None:
                continue
            img, lab = loaded  # img:(H,W,D,C) lab:(H,W,D)

            D = lab.shape[2]
            if z < 0 or z >= D:
                continue

            x_np = img[:, :, z, :]   # (H,W,C)
            y_np = lab[:, :, z]      # (H,W)

            x_t = _resize_img_tf(x_np, self.target_size)   # float32
            y_t = _resize_mask_tf(y_np, self.target_size)  # int32
            y_oh = _one_hot(y_t, self.num_classes)         # float32

            if self.do_augment:
                x_t, y_oh = _augment_pair_safe(x_t, y_oh, rrng)

            xb.append(x_t)
            yb.append(y_oh)

        if len(xb) == 0:
            x = tf.zeros((self.batch_size, self.target_size[0], self.target_size[1], 1), tf.float16)
            y = tf.zeros((self.batch_size, self.target_size[0], self.target_size[1], self.num_classes), tf.float32)
            return x, y

        while len(xb) < self.batch_size:
            xb.append(xb[-1])
            yb.append(yb[-1])

        x = tf.cast(tf.stack(xb, axis=0), tf.float16)
        y = tf.cast(tf.stack(yb, axis=0), tf.float32)
        return x, y


# ============================================================
# METRICS + LOSSES
# ============================================================
class DiceForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id: int, name="dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = int(class_id)
        self.intersection = self.add_weight(name="intersection", initializer="zeros", dtype=tf.float32)
        self.yt_sum       = self.add_weight(name="yt_sum", initializer="zeros", dtype=tf.float32)
        self.yp_sum       = self.add_weight(name="yp_sum", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true[..., self.class_id], tf.float32)
        yp = tf.cast(y_pred[..., self.class_id], tf.float32)
        self.intersection.assign_add(tf.reduce_sum(yt * yp))
        self.yt_sum.assign_add(tf.reduce_sum(yt))
        self.yp_sum.assign_add(tf.reduce_sum(yp))

    def result(self):
        eps = tf.constant(1e-6, tf.float32)
        return (2.0 * self.intersection + eps) / (self.yt_sum + self.yp_sum + eps)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.yt_sum.assign(0.0)
        self.yp_sum.assign(0.0)

class MeanDiceForeground(tf.keras.metrics.Metric):
    def __init__(self, num_classes=4, name="mean_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.ms = [DiceForClass(c, name=f"_dice_{c}") for c in range(1, self.num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for m in self.ms:
            m.update_state(y_true, y_pred)

    def result(self):
        return tf.reduce_mean([m.result() for m in self.ms])

    def reset_state(self):
        for m in self.ms:
            m.reset_state()

class IoUForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id: int, name="iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = int(class_id)
        self.intersection = self.add_weight(name="intersection", initializer="zeros", dtype=tf.float32)
        self.union        = self.add_weight(name="union", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true[..., self.class_id], tf.float32)
        yp = tf.cast(y_pred[..., self.class_id], tf.float32)
        inter = tf.reduce_sum(yt * yp)
        union = tf.reduce_sum(yt + yp - yt * yp)
        self.intersection.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        eps = tf.constant(1e-6, tf.float32)
        return (self.intersection + eps) / (self.union + eps)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

class MeanIoUForeground(tf.keras.metrics.Metric):
    def __init__(self, num_classes=4, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.ms = [IoUForClass(c, name=f"_iou_{c}") for c in range(1, self.num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for m in self.ms:
            m.update_state(y_true, y_pred)

    def result(self):
        return tf.reduce_mean([m.result() for m in self.ms])

    def reset_state(self):
        for m in self.ms:
            m.reset_state()

def _hd95_binary_np(y_true_bin, y_pred_bin):
    yt = y_true_bin.astype(np.uint8)
    yp = y_pred_bin.astype(np.uint8)

    if yt.sum() == 0 and yp.sum() == 0:
        return 0.0
    if yt.sum() == 0 or yp.sum() == 0:
        return 95.0

    dt_true = distance_transform_edt(1 - yt)
    dt_pred = distance_transform_edt(1 - yp)

    sds = np.concatenate([dt_true[yp.astype(bool)], dt_pred[yt.astype(bool)]])
    return float(np.percentile(sds, 95))

class HD95ForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id: int, name="hd95", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = int(class_id)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.argmax(y_true, axis=-1)
        yp = tf.argmax(y_pred, axis=-1)

        def _batch_hd95(yt_b, yp_b):
            yt_b = np.asarray(yt_b)
            yp_b = np.asarray(yp_b)
            vals = []
            for i in range(yt_b.shape[0]):
                yt_bin = (yt_b[i] == self.class_id).astype(np.uint8)
                yp_bin = (yp_b[i] == self.class_id).astype(np.uint8)
                vals.append(_hd95_binary_np(yt_bin, yp_bin))
            return np.asarray(vals, dtype=np.float32)

        hd = tf.py_function(_batch_hd95, [yt, yp], Tout=tf.float32)
        hd = tf.reshape(hd, [-1])
        self.total.assign_add(tf.reduce_sum(hd))
        self.count.assign_add(tf.cast(tf.size(hd), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class MeanHD95Foreground(tf.keras.metrics.Metric):
    def __init__(self, num_classes=4, name="mean_hd95", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.ms = [HD95ForClass(c, name=f"_hd95_{c}") for c in range(1, self.num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for m in self.ms:
            m.update_state(y_true, y_pred)

    def result(self):
        return tf.reduce_mean([m.result() for m in self.ms])

    def reset_state(self):
        for m in self.ms:
            m.reset_state()

def tversky_multiclass(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6, exclude_bg=True):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    axes = [0, 1, 2]
    tp = tf.reduce_sum(y_true * y_pred, axis=axes)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=axes)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=axes)

    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    if exclude_bg:
        t = t[1:]
    return 1.0 - tf.reduce_mean(t)

def ce_tversky_multiclass(lam=0.7, alpha=0.3, beta=0.7, exclude_bg=True):
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        return (1.0 - lam) * ce(y_true_f, y_pred_f) + lam * tversky_multiclass(
            y_true_f, y_pred_f, alpha=alpha, beta=beta, exclude_bg=exclude_bg
        )
    return loss

def ce_dice_multiclass(lam=0.5, exclude_bg=True):
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        axes = [0, 1, 2]
        inter = tf.reduce_sum(y_true * y_pred, axis=axes)
        denom = tf.reduce_sum(y_true + y_pred, axis=axes)
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        if exclude_bg:
            dice = dice[1:]
        dice_loss = 1.0 - tf.reduce_mean(dice)
        return (1.0 - lam) * ce(y_true, y_pred) + lam * dice_loss
    return loss

# ============================================================
# BUILD CASE LISTS + SEQUENCES + SANITY CHECKS
# ============================================================
print("\n--- Raw .h5 ---")
print("TRAIN:", len(_list_h5_files(TRAIN_H5_DIR)))
print("VAL  :", len(_list_h5_files(VAL_H5_DIR)))

train_cases = build_acdc_case_list(TRAIN_H5_DIR, require_label=True)
val_cases   = build_acdc_case_list(VAL_H5_DIR,   require_label=True)

train_names = [c["name"] for c in train_cases]
val_names   = [c["name"] for c in val_cases]

print("\n--- Valid cases ---")
print("TRAIN:", len(train_names))
print("VAL  :", len(val_names))
print("Overlap:", len(set(train_names) & set(val_names)))

if len(train_names) < 1:
    raise ValueError("No training cases found. Check TRAIN_H5_DIR.")
if len(val_names) < 1:
    raise ValueError("No validation cases found. Check VAL_H5_DIR labels exist.")

train_seq = ACDCCaseWiseSliceSequence(
    cases=train_cases,
    case_names=train_names,
    target_size=TARGET_SIZE,
    batch_size=TRAIN_BATCH,
    cases_per_epoch=min(TRAIN_CASES_PER_EPOCH, len(train_names)),
    k_rv=TRAIN_K_RV, k_myo=TRAIN_K_MYO, k_lv=TRAIN_K_LV, k_neg=TRAIN_K_NEG,
    num_classes=NUM_CLASSES,
    normalize=NORMALIZE,
    shuffle_cases=True,
    seed=SEED,
    freeze_slices=False,
    cache_cases=CACHE_CASES,
    do_augment=DO_AUGMENT_TRAIN,
)

val_seq = ACDCCaseWiseSliceSequence(
    cases=val_cases,
    case_names=val_names,
    target_size=TARGET_SIZE,
    batch_size=VAL_BATCH,
    cases_per_epoch=min(VAL_CASES_PER_EPOCH, len(val_names)),
    k_rv=VAL_K_RV, k_myo=VAL_K_MYO, k_lv=VAL_K_LV, k_neg=VAL_K_NEG,
    num_classes=NUM_CLASSES,
    normalize=NORMALIZE,
    shuffle_cases=False,
    seed=999,
    freeze_slices=VAL_FREEZE_SLICES,
    cache_cases=CACHE_CASES,
    do_augment=False,
)

print("\nTrain steps/epoch:", len(train_seq), "| Val steps/epoch:", len(val_seq))
print("Expected train slices/epoch:", train_seq.cases_per_epoch * (TRAIN_K_RV + TRAIN_K_MYO + TRAIN_K_LV + TRAIN_K_NEG))
print("Expected val slices/epoch:",   val_seq.cases_per_epoch   * (VAL_K_RV + VAL_K_MYO + VAL_K_LV + VAL_K_NEG))

def _batch_label_sums(y_onehot):
    return tf.reduce_sum(y_onehot, axis=(0,1,2)).numpy()

x0, y0 = train_seq[0]
print("\n--- Sanity batch0 (train) ---")
print("x0:", x0.shape, x0.dtype, "range:", float(tf.reduce_min(x0)), float(tf.reduce_max(x0)))
print("y0 unique:", np.unique(tf.argmax(y0, axis=-1).numpy()))
print("class sums [bg,rv,myo,lv]:", _batch_label_sums(y0))

xv, yv = val_seq[0]
print("\n--- Sanity batch0 (val) ---")
print("xv:", xv.shape, xv.dtype, "range:", float(tf.reduce_min(xv)), float(tf.reduce_max(xv)))
print("yv unique:", np.unique(tf.argmax(yv, axis=-1).numpy()))
print("class sums [bg,rv,myo,lv]:", _batch_label_sums(yv))

print("\n Pipeline ready: train_seq, val_seq")

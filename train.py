import os, json
import tensorflow as tf


import random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# optional best-effort determinism (safe to leave; may be ignored on some ops/GPUs)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# ------------------------------------------------------------
# SAVE ROOT (CHECK + USE)
# ------------------------------------------------------------
SAVE_ROOT = "/data2/farhana/datasets/ACDC/ACDC_preprocessed"
CKPT_DIR  = os.path.join(SAVE_ROOT, "checkpoints")
LOG_DIR   = os.path.join(SAVE_ROOT, "logs")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("SAVE_ROOT:", SAVE_ROOT, "| exists:", os.path.isdir(SAVE_ROOT))
print("CKPT_DIR :", CKPT_DIR)
print("LOG_DIR  :", LOG_DIR)

if not os.path.isdir(SAVE_ROOT):
    raise FileNotFoundError(f"❌ SAVE_ROOT not found: {SAVE_ROOT}")

# ------------------------------------------------------------
# PHASE CONFIG
# ------------------------------------------------------------
WARMUP_EPOCHS = 30
MAIN_EPOCHS   = 300

CKPT_PATH   = os.path.join(CKPT_DIR, "acdc_weight_last5.keras")
WARMUP_JSON = os.path.join(LOG_DIR,  "warmup_epoch_last5.json")
MAIN_JSON   = os.path.join(LOG_DIR,  "main_epoch_last5.json")

WARMUP_CSV  = os.path.join(LOG_DIR,  "train_metrics_warmup_last5.csv")
MAIN_CSV    = os.path.join(LOG_DIR,  "train_metrics_main_last5.csv")

DELETE_OLD = True  # True = restart everything

if DELETE_OLD:
    for f in [CKPT_PATH, WARMUP_JSON, MAIN_JSON, WARMUP_CSV, MAIN_CSV]:
        if os.path.exists(f):
            os.remove(f)
            print(" Deleted:", f)

# ------------------------------------------------------------
# Epoch tracker helpers
# ------------------------------------------------------------
def _load_epoch(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return int(json.load(f).get("epoch", 0))
        except Exception:
            return 0
    return 0

def _save_epoch(path, epoch):
    with open(path, "w") as f:
        json.dump({"epoch": int(epoch)}, f)

class PhaseEpochSaver(tf.keras.callbacks.Callback):
    def __init__(self, json_path):
        super().__init__()
        self.json_path = json_path

    def on_epoch_end(self, epoch, logs=None):
        _save_epoch(self.json_path, epoch + 1)

# ------------------------------------------------------------
# Build / Load model
# ------------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(1e-4)

custom_map = {
    "WeightedSum2": WeightedSum2,
    "Custom>WeightedSum2": WeightedSum2,
    "Custom.WeightedSum2": WeightedSum2,
}

if os.path.exists(CKPT_PATH):
    print("\n✅ Loading checkpoint:", CKPT_PATH)
    model = tf.keras.models.load_model(
        CKPT_PATH,
        custom_objects=custom_map,
        compile=False,
        safe_mode=False,
    )
else:
    print("\n New model")
    model = model_arch(input_shape=(256, 256, 1), num_classes=4, base_filters=32,
                            kernel_size=3, activation="relu", norm_type="batch",
                            conv_type="separableconv2d", pooling_type="maxpool2d", fusion_method="add", dropout_deep=0.2, multilabel=False,)

# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------
model.compile(
    optimizer=optimizer,
    loss=ce_tversky_multiclass(lam=0.7, alpha=0.3, beta=0.7, exclude_bg=True),
    metrics=[
        DiceForClass(1, name="dice_rv"),
        DiceForClass(2, name="dice_myo"),
        DiceForClass(3, name="dice_lv"),
        MeanDiceForeground(num_classes=4, name="mean_dice"),
        IoUForClass(1, name="iou_rv"),
        IoUForClass(2, name="iou_myo"),
        IoUForClass(3, name="iou_lv"),
        MeanIoUForeground(num_classes=4, name="mean_iou"),
        HD95ForClass(1, name="hd95_rv"),
        HD95ForClass(2, name="hd95_myo"),
        HD95ForClass(3, name="hd95_lv"),
        MeanHD95Foreground(num_classes=4, name="mean_hd95"),
    ],
    jit_compile=False,
)

print("\nsteps_per_epoch =", len(train_seq), "| validation_steps =", len(val_seq))

# ------------------------------------------------------------
# Phase state
# ------------------------------------------------------------
warmup_done = _load_epoch(WARMUP_JSON)   # 0..20
main_done   = _load_epoch(MAIN_JSON)     # 0..300

print("\nProgress:")
print(f"  Warmup: {warmup_done}/{WARMUP_EPOCHS}")
print(f"  Main  : {main_done}/{MAIN_EPOCHS}")

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
# MAIN checkpoint only
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    CKPT_PATH,
    monitor="val_mean_dice",
    mode="max",
    save_best_only=True,
    verbose=1,
)

# CSV loggers (append=True so resume keeps growing)
warmup_csv_cb = tf.keras.callbacks.CSVLogger(WARMUP_CSV, append=True)
main_csv_cb   = tf.keras.callbacks.CSVLogger(MAIN_CSV, append=True)

# ============================================================
# PHASE 1 — WARMUP (NO CHECKPOINT SAVING)
# ============================================================
if warmup_done < WARMUP_EPOCHS:
    print("\n================= PHASE 1: WARMUP (NO CKPT) =================")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=WARMUP_EPOCHS,
        initial_epoch=warmup_done,
        callbacks=[PhaseEpochSaver(WARMUP_JSON), warmup_csv_cb],
        verbose=1,
    )
    warmup_done = _load_epoch(WARMUP_JSON)
    print(f" Warmup finished: {warmup_done}/{WARMUP_EPOCHS}")
else:
    print("\n Warmup already done. Skipping.")

# ============================================================
# PHASE 2 — MAIN (SAVE + SHOW EPOCH 1..300 IN CONSOLE)
#   Trick: we run epochs=(main_done + remaining) but print is
#   controlled by initial_epoch, so it will display 1..300.
# ============================================================
if main_done < MAIN_EPOCHS:
    print("\n================= PHASE 2: MAIN (CKPT ON) =================")

    # We want console to show Epoch (main_done+1) -> 300, but labeled 1..300 overall.
    # So we simply run epochs=MAIN_EPOCHS and initial_epoch=main_done.
    # This naturally displays Epoch {main_done+1}/{MAIN_EPOCHS} ... {MAIN_EPOCHS}/{MAIN_EPOCHS}
    print(f"Logging will show: Epoch {main_done+1}/{MAIN_EPOCHS} ... {MAIN_EPOCHS}/{MAIN_EPOCHS}")

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=MAIN_EPOCHS,
        initial_epoch=main_done,
        callbacks=[ckpt_cb, PhaseEpochSaver(MAIN_JSON), main_csv_cb],
        verbose=1,
    )

    main_done = _load_epoch(MAIN_JSON)
    print(f" Main finished: {main_done}/{MAIN_EPOCHS}")
else:
    print("\n Main already completed.")

print("\n DONE")
print("Best model :", CKPT_PATH)
print("Warmup JSON:", WARMUP_JSON)
print("Main JSON  :", MAIN_JSON)
print("Warmup CSV :", WARMUP_CSV)
print("Main CSV   :", MAIN_CSV)
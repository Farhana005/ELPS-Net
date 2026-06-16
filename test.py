import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt

# ============================================================
# CONFIG
# ============================================================

CKPT_PATH = "/data2/farhana/datasets/ACDC/ACDC_preprocessed/checkpoints/acdc_weight_last4.keras"

FALLBACK_CKPT = "/mnt/data2/farhana/datasets/ACDC/ACDC_preprocessed/checkpoints/acdc_weight_last4.keras"  # uploaded file
SEED = 42
TARGET_SIZE = (256, 256)
NUM_CLASSES = 4  # 0=BG,1=RV,2=MYO,3=LV
ALPHA = 0.9

# evaluation controls
MAX_FG_SLICES_PER_CASE = 12   # speed: evaluate up to K FG slices/case (None for all FG)
VIS_RANDOM_CASES = 10         # show random cases after full evaluation

# ============================================================
# REQUIRE val_cases + helpers exist (from your ACDC pipeline cell)
# ============================================================
for need in ["val_cases", "_safe_read_h5", "_to_dhw", "_to_label_dhw", "_minmax"]:
    if need not in globals():
        raise ValueError(f"'{need}' is not defined. Run your ACDC data pipeline cell first.")

# ============================================================
# LOAD MODEL
# ============================================================
model_path = CKPT_PATH if os.path.exists(CKPT_PATH) else FALLBACK_CKPT
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at:\n- {CKPT_PATH}\n- {FALLBACK_CKPT}")

model = tf.keras.models.load_model(
    model_path,
    custom_objects={"WeightedSum2": WeightedSum2},
    compile=False,
    # safe_mode=False,
)
print(" Loaded model:", model_path)
print("Model input shape:", model.input_shape)



# ============================================================
# TF inference wrapper (avoid retracing)
# ============================================================
@tf.function(reduce_retracing=True)
def infer(x):
    return model(x, training=False)

def resize_img_tf(x):
    return tf.image.resize(x, TARGET_SIZE, method="bilinear")

def resize_lab_tf(y):
    y = tf.image.resize(y[..., None], TARGET_SIZE, method="nearest")
    return tf.cast(tf.squeeze(y, -1), tf.int32)

def load_case(case):
    img_np, lab_np = _safe_read_h5(case["path"])
    img_np = _to_dhw(img_np)
    lab_np = _to_label_dhw(lab_np)
    if img_np is None or lab_np is None:
        return None, None

    # img -> (H,W,D,1/C), lab -> (H,W,D)
    if img_np.ndim == 3:
        img = np.transpose(img_np, (1, 2, 0))[..., None].astype(np.float32)
    else:
        img = np.transpose(img_np, (1, 2, 0, 3)).astype(np.float32)

    lab = np.transpose(lab_np, (1, 2, 0)).astype(np.uint8)

    img = _minmax(img)
    return img, lab

# ============================================================
# Basic Metrics
# ============================================================
def dice_binary(gtb, prb, eps=1e-6):
    gtb = gtb.astype(bool); prb = prb.astype(bool)
    inter = np.logical_and(gtb, prb).sum()
    return (2.0 * inter + eps) / (gtb.sum() + prb.sum() + eps)

def iou_binary(gtb, prb, eps=1e-6):
    gtb = gtb.astype(bool); prb = prb.astype(bool)
    inter = np.logical_and(gtb, prb).sum()
    union = np.logical_or(gtb, prb).sum()
    return (inter + eps) / (union + eps)

def hd95_binary(gtb, prb):
    gtb = gtb.astype(bool); prb = prb.astype(bool)
    if gtb.sum() == 0 and prb.sum() == 0:
        return 0.0
    if gtb.sum() == 0 or prb.sum() == 0:
        return 95.0
    dt_gt = distance_transform_edt(~gtb)
    dt_pr = distance_transform_edt(~prb)
    sds = np.concatenate([dt_gt[prb], dt_pr[gtb]])
    return float(np.percentile(sds, 95))

def slice_metrics(gt, pr):
    per = {}
    for c in [1,2,3]:
        gtb = (gt == c)
        prb = (pr == c)
        per[c] = (dice_binary(gtb, prb), iou_binary(gtb, prb), hd95_binary(gtb, prb), int(gtb.sum()))
    # mean over GT-present classes only
    ds=[]; is_=[]; hs=[]
    for c in [1,2,3]:
        if per[c][3] > 0:
            ds.append(per[c][0]); is_.append(per[c][1]); hs.append(per[c][2])
    if len(ds)==0:
        mean_fg = (1.0, 1.0, 0.0)
    else:
        mean_fg = (float(np.mean(ds)), float(np.mean(is_)), float(np.mean(hs)))
    return per, mean_fg


# ============================================================
# Evaluate ALL validation cases (case-wise average over FG slices)
# ============================================================
rng = np.random.default_rng(SEED)

case_rows = []

print("\n🔎 Evaluating ALL validation cases:", len(val_cases))

for idx, case in enumerate(val_cases):
    name = case["name"]
    img, lab = load_case(case)
    if img is None:
        continue

    mx = lab.max(axis=(0,1))
    fg_slices = np.where(mx > 0)[0].tolist()
    if len(fg_slices) == 0:
        continue

    if (MAX_FG_SLICES_PER_CASE is not None) and (len(fg_slices) > MAX_FG_SLICES_PER_CASE):
        fg_slices = list(rng.choice(fg_slices, size=MAX_FG_SLICES_PER_CASE, replace=False))
        fg_slices = sorted(fg_slices)

    per_class_acc = {1: [], 2: [], 3: []}   # (dice,iou,hd95) GT-present slices
    mean_fg_list = []


    for z in fg_slices:
        x = resize_img_tf(img[:, :, z, :])                   # (H,W,C) float32
        gt = resize_lab_tf(lab[:, :, z]).numpy().astype(np.uint8)

        x_in = tf.cast(x[None, ...], tf.float16)            # (1,H,W,C)
        probs = infer(x_in)[0].numpy().astype(np.float32)   # (H,W,K)
        pr = np.argmax(probs, axis=-1).astype(np.uint8)

        per, mean_fg = slice_metrics(gt, pr)
        mean_fg_list.append(mean_fg)

        for c in [1,2,3]:
            d,iou,hd,gtpx = per[c]
            if gtpx > 0:
                per_class_acc[c].append((d,iou,hd))

    if len(mean_fg_list) == 0:
        continue

    mean_fg_arr = np.array(mean_fg_list, dtype=np.float32)
    case_mean_dice = float(mean_fg_arr[:,0].mean())
    case_mean_iou  = float(mean_fg_arr[:,1].mean())
    case_mean_hd95 = float(mean_fg_arr[:,2].mean())

    def pc_mean(c, k):
        a = per_class_acc[c]
        if len(a)==0:
            return np.nan
        return float(np.array(a, dtype=np.float32)[:,k].mean())

    row = [
        name,
        case_mean_dice, case_mean_iou, case_mean_hd95,
        pc_mean(1,0), pc_mean(2,0), pc_mean(3,0),
        pc_mean(1,1), pc_mean(2,1), pc_mean(3,1),
        pc_mean(1,2), pc_mean(2,2), pc_mean(3,2),
        len(fg_slices)
    ]
    case_rows.append(row)



# ============================================================
# Print summary (Dice/IoU/HD95) + class-wise mean/std
# ============================================================
if len(case_rows) == 0:
    raise ValueError("No valid labeled FG cases evaluated. Check val_cases / labels.")

case_rows = sorted(case_rows, key=lambda r: r[1])  # sort by mean Dice ascending
vals = np.array([[r[1], r[2], r[3]] for r in case_rows], dtype=np.float32)

print("\n================= OVERALL (case-wise) =================")
print("Cases evaluated:", len(case_rows))
print("MeanFG Dice (mean±std): %.4f ± %.4f" % (float(vals[:,0].mean()), float(vals[:,0].std())))
print("MeanFG IoU  (mean±std): %.4f ± %.4f" % (float(vals[:,1].mean()), float(vals[:,1].std())))
print("MeanFG HD95 (mean±std): %.2f ± %.2f" % (float(vals[:,2].mean()), float(vals[:,2].std())))

def nanmean(xs):
    xs = np.array(xs, dtype=np.float32)
    return float(np.nanmean(xs))

def nanstd(xs):
    xs = np.array(xs, dtype=np.float32)
    return float(np.nanstd(xs))

print("\nPer-class (case-wise mean±std over GT-present slices):")
print("  Dice: RV=%.4f±%.4f | MYO=%.4f±%.4f | LV=%.4f±%.4f" % (
    nanmean([r[4] for r in case_rows]), nanstd([r[4] for r in case_rows]),
    nanmean([r[5] for r in case_rows]), nanstd([r[5] for r in case_rows]),
    nanmean([r[6] for r in case_rows]), nanstd([r[6] for r in case_rows]),
))
print("  IoU : RV=%.4f±%.4f | MYO=%.4f±%.4f | LV=%.4f±%.4f" % (
    nanmean([r[7] for r in case_rows]), nanstd([r[7] for r in case_rows]),
    nanmean([r[8] for r in case_rows]), nanstd([r[8] for r in case_rows]),
    nanmean([r[9] for r in case_rows]), nanstd([r[9] for r in case_rows]),
))
print("  HD95: RV=%.2f±%.2f | MYO=%.2f±%.2f | LV=%.2f±%.2f" % (
    nanmean([r[10] for r in case_rows]), nanstd([r[10] for r in case_rows]),
    nanmean([r[11] for r in case_rows]), nanstd([r[11] for r in case_rows]),
    nanmean([r[12] for r in case_rows]), nanstd([r[12] for r in case_rows]),
))

print("\nWorst 5 cases by MeanFG Dice:")
for r in case_rows[:5]:
    print(f"  {r[0]} | Dice={r[1]:.4f} IoU={r[2]:.4f} HD95={r[3]:.2f} | FG_slices_eval={r[13]}")
print("\nBest 5 cases by MeanFG Dice:")
for r in case_rows[-5:][::-1]:
    print(f"  {r[0]} | Dice={r[1]:.4f} IoU={r[2]:.4f} HD95={r[3]:.2f} | FG_slices_eval={r[13]}")


# ============================================================
# Track Best Case for Each Class (RV, MYO, LV) by Dice, IoU, and HD95
# ============================================================
best_case_classwise = {
    "RV":  {"dice": {"name": None, "val": -np.inf}, "iou": {"name": None, "val": -np.inf}, "hd95": {"name": None, "val": np.inf}},
    "MYO": {"dice": {"name": None, "val": -np.inf}, "iou": {"name": None, "val": -np.inf}, "hd95": {"name": None, "val": np.inf}},
    "LV":  {"dice": {"name": None, "val": -np.inf}, "iou": {"name": None, "val": -np.inf}, "hd95": {"name": None, "val": np.inf}},
}

for r in case_rows:
    name = r[0]
    dice_rv, dice_myo, dice_lv = r[4], r[5], r[6]
    iou_rv,  iou_myo,  iou_lv  = r[7], r[8], r[9]
    hd_rv,   hd_myo,   hd_lv   = r[10], r[11], r[12]

    if not np.isnan(dice_rv) and dice_rv > best_case_classwise["RV"]["dice"]["val"]:
        best_case_classwise["RV"]["dice"] = {"name": name, "val": float(dice_rv)}
    if not np.isnan(iou_rv) and iou_rv > best_case_classwise["RV"]["iou"]["val"]:
        best_case_classwise["RV"]["iou"] = {"name": name, "val": float(iou_rv)}
    if not np.isnan(hd_rv) and hd_rv < best_case_classwise["RV"]["hd95"]["val"]:
        best_case_classwise["RV"]["hd95"] = {"name": name, "val": float(hd_rv)}

    if not np.isnan(dice_myo) and dice_myo > best_case_classwise["MYO"]["dice"]["val"]:
        best_case_classwise["MYO"]["dice"] = {"name": name, "val": float(dice_myo)}
    if not np.isnan(iou_myo) and iou_myo > best_case_classwise["MYO"]["iou"]["val"]:
        best_case_classwise["MYO"]["iou"] = {"name": name, "val": float(iou_myo)}
    if not np.isnan(hd_myo) and hd_myo < best_case_classwise["MYO"]["hd95"]["val"]:
        best_case_classwise["MYO"]["hd95"] = {"name": name, "val": float(hd_myo)}

    if not np.isnan(dice_lv) and dice_lv > best_case_classwise["LV"]["dice"]["val"]:
        best_case_classwise["LV"]["dice"] = {"name": name, "val": float(dice_lv)}
    if not np.isnan(iou_lv) and iou_lv > best_case_classwise["LV"]["iou"]["val"]:
        best_case_classwise["LV"]["iou"] = {"name": name, "val": float(iou_lv)}
    if not np.isnan(hd_lv) and hd_lv < best_case_classwise["LV"]["hd95"]["val"]:
        best_case_classwise["LV"]["hd95"] = {"name": name, "val": float(hd_lv)}

print("\n================= BEST CASES CLASS-WISE =================")
for class_name in ["RV", "MYO", "LV"]:
    print(f"{class_name}:")
    print(f"  Best Dice: {best_case_classwise[class_name]['dice']['name']} | {best_case_classwise[class_name]['dice']['val']:.4f}")
    print(f"  Best IoU : {best_case_classwise[class_name]['iou']['name']}  | {best_case_classwise[class_name]['iou']['val']:.4f}")
    print(f"  Best HD95: {best_case_classwise[class_name]['hd95']['name']} | {best_case_classwise[class_name]['hd95']['val']:.2f}")

# ============================================================
# Visualize random cases 
# ============================================================
rng = np.random.default_rng(SEED + 999)
pick_n = min(VIS_RANDOM_CASES, len(case_rows))
vis_names = [case_rows[i][0] for i in rng.choice(len(case_rows), size=pick_n, replace=False)]

name_to_case = {c["name"]: c for c in val_cases}

def pick_rep_fg_slice(lab_hw_d):
    mx = lab_hw_d.max(axis=(0,1))
    fg = np.where(mx > 0)[0].tolist()
    if len(fg)==0:
        return int(lab_hw_d.shape[2]//2)
    return int(rng.choice(fg))

print("\n================= SHOWING RANDOM CASES =================")
for j, name in enumerate(vis_names, start=1):
    case = name_to_case[name]
    img, lab = load_case(case)
    if img is None:
        continue
    z = pick_rep_fg_slice(lab)

    x = resize_img_tf(img[:, :, z, :])
    gt = resize_lab_tf(lab[:, :, z]).numpy().astype(np.uint8)

    x_in = tf.cast(x[None, ...], tf.float16)
    probs = infer(x_in)[0].numpy().astype(np.float32)
    pr = np.argmax(probs, axis=-1).astype(np.uint8)

    per, mean_fg = slice_metrics(gt, pr)

    img2d = x[..., 0].numpy()
    img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-6)

    print(f"\n[{j}/{pick_n}] Case={name} | Slice={z} | MeanFG Dice={mean_fg[0]:.4f} IoU={mean_fg[1]:.4f} HD95={mean_fg[2]:.2f}")


    cols = 4
    fig, ax = plt.subplots(1, cols, figsize=(3.6*cols, 4))
    fig.suptitle(
        f"Case: {name} | Slice: {z} | MeanFG Dice={mean_fg[0]:.3f}  IoU={mean_fg[1]:.3f}  HD95={mean_fg[2]:.1f}",
        fontsize=12
    )

    ax[0].imshow(img2d, cmap="gray"); ax[0].set_title("Image"); ax[0].axis("off")
    ax[1].imshow(gt, cmap=seg_cmap, vmin=0, vmax=3); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(pr, cmap=seg_cmap, vmin=0, vmax=3); ax[2].set_title("Pred"); ax[2].axis("off")
    ax[3].imshow(img2d, cmap="gray")
    ax[3].imshow(pr, cmap=seg_cmap, vmin=0, vmax=3, alpha=ALPHA)
    ax[3].set_title(f"Overlay α={ALPHA}"); ax[3].axis("off")


    plt.tight_layout()
    plt.show()
    
# ============================================================
# Create 'best' variable for visualization cell
# ============================================================
# Create best list: sort cases by Dice score and keep top 5
best = []
for r in case_rows[-10:][::-1]:  # Get top 5 cases (already sorted by Dice)
    name = r[0]
    dice_val = r[1]
    # Find the corresponding case dictionary from val_cases
    case_dict = next((c for c in val_cases if c["name"] == name), {"name": name})
    best.append((name, dice_val, case_dict))

print(f"\n Created 'best' variable with {len(best)} top cases")

print("\n Done. (Full validation evaluated + random visualizations + class-wise mean/std)")

print("\n Done. (Full validation evaluated + random visualizations + class-wise mean/std)")

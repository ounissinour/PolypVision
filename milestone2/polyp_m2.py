"""Milestone 2 - retrain Polyp-PVT with a proper train / validation / test protocol.

The official Polyp-PVT `Train.py` scores the five TEST sets after every epoch and keeps the epoch
with the best test Dice, so the reported model is chosen on the test data. Here the model is chosen
on a held-out VALIDATION split of TrainDataset (made by milestone1/prepare_data.py) and the test
sets are touched once, at the end, with the chosen checkpoint.

What this module provides (used by M2_Improve.ipynb, which fetches it like M1 fetches prepare_data.py):

  AUGMENTATIONS  none | flip | flip_rotate (the official option) | strong
  LOSSES         structure_official (what the official code really computes) | structure_weighted
                 (the weighted BCE + weighted IoU the paper describes) | bce | dice | bce_dice
  Config         one training run (loss, augmentation, image size, epochs, ...)
  train()        trains one run, keeps the best epoch on validation mean Dice, resumes after a
                 Colab disconnect, then scores the chosen checkpoint once on the five test sets
  evaluate()     prediction maps exactly as the official Test.py + mean Dice / IoU exactly as
                 Milestone 1 (pysodmetrics, 256 thresholds, = the papers' MATLAB toolbox)
  collect()      one table with every finished run
  MedSAM helpers boxes_from_mask(), load_medsam(), evaluate_medsam() for the optional MedSAM test
"""

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from py_sod_metrics import DICEHandler, FmeasureV2, IOUHandler
from torch.utils.data import DataLoader, Dataset

TEST_SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ============================================================================ images

def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_input(rgb, size):
    """Resize and normalise like the official loaders (PIL bilinear Resize, ToTensor, Normalize)."""
    image = Image.fromarray(np.ascontiguousarray(rgb)).resize((size, size), Image.BILINEAR)
    x = (np.asarray(image, np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# ============================================================================ augmentations
# Each step works on the full-resolution uint8 image (H x W x 3, RGB) and its 0/255 mask (H x W),
# before the resize to the training size. `rng` is a numpy Generator.

def _flip(image, mask, rng):
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    return image, mask


def _rotate(image, mask, rng, max_degrees=90):
    """Random rotation in [-90, 90] degrees with black corners, like the official RandomRotation(90)."""
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-max_degrees, max_degrees), 1.0)
    image = cv2.warpAffine(np.ascontiguousarray(image), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return image, mask


def _scale_crop(image, mask, rng, min_side=0.7, tries=5):
    """Random crop of 70-100 % of each side (a zoom-in); keeps some polyp in the crop if it can."""
    h, w = mask.shape
    for _ in range(tries):
        s = rng.uniform(min_side, 1.0)
        ch, cw = int(h * s), int(w * s)
        y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        if mask[y:y + ch, x:x + cw].any() or not mask.any():
            return image[y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]
    return image, mask


def _color(image, rng):
    """Small brightness / contrast / saturation / hue changes (lighting differs between hospitals)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] *= rng.uniform(0.8, 1.2)
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    image = (image - image.mean()) * rng.uniform(0.8, 1.2) + image.mean()
    return np.clip(image, 0, 255).astype(np.uint8)


def _blur(image, rng):
    """Gaussian blur on 30 % of the images (motion / focus blur in colonoscopy video)."""
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(np.ascontiguousarray(image), (k, k), 0)
    return image


def _aug_none(image, mask, rng):
    return image, mask


def _aug_flip(image, mask, rng):
    return _flip(image, mask, rng)


def _aug_flip_rotate(image, mask, rng):
    return _rotate(*_flip(image, mask, rng), rng)


def _aug_strong(image, mask, rng):
    image, mask = _scale_crop(image, mask, rng)
    image, mask = _rotate(*_flip(image, mask, rng), rng)
    return _blur(_color(image, rng), rng), mask


AUGMENTATIONS = {
    "none": _aug_none,                  # official default (Train.py --augmentation False)
    "flip": _aug_flip,                  # horizontal + vertical flips
    "flip_rotate": _aug_flip_rotate,    # the official --augmentation True: flips + rotation up to 90 deg
    "strong": _aug_strong,              # flip_rotate + zoom-in crop + colour jitter + blur
}


# ============================================================================ datasets

class TrainSet(Dataset):
    """(image, mask) at `size` x `size`. Image: 3 x H x W, ImageNet-normalised. Mask: 1 x H x W in
    [0, 1], bilinearly resized like the official loader (soft edges)."""

    def __init__(self, rows, data_dir, size, augmentation="none"):
        if augmentation not in AUGMENTATIONS:
            raise ValueError(f"unknown augmentation {augmentation!r}, choose from {list(AUGMENTATIONS)}")
        data_dir = Path(data_dir)
        self.items = [(data_dir / r.image, data_dir / r.mask) for r in rows.itertuples()]
        self.size = size
        self.augment = AUGMENTATIONS[augmentation]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # one generator per sample, seeded from torch so every DataLoader worker draws differently
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        image_path, mask_path = self.items[i]
        image = read_rgb(image_path)
        mask = np.where(np.array(Image.open(mask_path).convert("L")) > 127, 255, 0).astype(np.uint8)
        image, mask = self.augment(image, mask, rng)
        x = to_input(image, self.size)
        m = Image.fromarray(np.ascontiguousarray(mask)).resize((self.size, self.size), Image.BILINEAR)
        y = torch.from_numpy(np.asarray(m, np.float32) / 255)[None]
        return x, y


# ============================================================================ losses
# All take raw logits (B x 1 x H x W) and a mask in [0, 1] and return a scalar.

def _edge_weight(mask):
    """Polyp-PVT / PraNet pixel weight: 1 near the polyp boundary up to 6, 1 elsewhere."""
    return 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)


def _weighted_iou(logits, mask, weight):
    p = torch.sigmoid(logits)
    inter = (p * mask * weight).sum(dim=(2, 3))
    union = ((p + mask) * weight).sum(dim=(2, 3))
    return 1 - (inter + 1) / (union - inter + 1)


def structure_official(logits, mask):
    """What the official code computes. It calls binary_cross_entropy_with_logits(..., reduce='none');
    `reduce` is a deprecated boolean argument and the string 'none' counts as True, so PyTorch returns
    the MEAN BCE (a scalar). Weighting that scalar by the edge weight and normalising gives it back
    unchanged, so the 'weighted BCE' is in fact a plain BCE. The weighted IoU part works as described."""
    weight = _edge_weight(mask)
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    return (bce + _weighted_iou(logits, mask, weight)).mean()


def structure_weighted(logits, mask):
    """The structure loss as the PraNet / Polyp-PVT papers describe it: per-pixel edge-weighted BCE
    + edge-weighted IoU."""
    weight = _edge_weight(mask)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    return (wbce + _weighted_iou(logits, mask, weight)).mean()


def bce(logits, mask):
    return F.binary_cross_entropy_with_logits(logits, mask)


def dice(logits, mask):
    """Soft Dice loss per image, averaged over the batch."""
    p = torch.sigmoid(logits)
    inter = (p * mask).sum(dim=(2, 3))
    return (1 - (2 * inter + 1) / (p.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) + 1)).mean()


def bce_dice(logits, mask):
    return bce(logits, mask) + dice(logits, mask)


LOSSES = {
    "structure_official": structure_official,
    "structure_weighted": structure_weighted,
    "bce": bce,
    "dice": dice,
    "bce_dice": bce_dice,
}


# ============================================================================ model

def build_model(pvt_repo):
    """PolypPVT from the official repository. Its constructor loads ./pretrained_pth/pvt_v2_b2.pth
    (ImageNet PVTv2-B2 backbone) relative to the working directory, so build it from inside the repo."""
    pvt_repo = str(Path(pvt_repo).resolve())
    if pvt_repo not in sys.path:
        sys.path.insert(0, pvt_repo)
    from lib.pvt import PolypPVT

    cwd = os.getcwd()
    os.chdir(pvt_repo)
    try:
        return PolypPVT()
    finally:
        os.chdir(cwd)


# ============================================================================ evaluation

@torch.no_grad()
def predict_map(model, rgb, size, out_hw, device):
    """One prediction map, the official Test.py way: resize to size, P1 + P2, bilinear upsample to
    the mask size, sigmoid, min-max normalise, x255 (cv2.imwrite rounds the same way)."""
    x = to_input(rgb, size)[None].to(device)
    p1, p2 = model(x)
    res = F.interpolate(p1 + p2, size=tuple(out_hw), mode="bilinear", align_corners=False)
    res = res.sigmoid().squeeze().float().cpu().numpy()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return np.clip(np.rint(res * 255), 0, 255).astype(np.uint8)


class Scorer:
    """Mean Dice / mean IoU exactly as in Milestone 1 (pysodmetrics FmeasureV2, 256 thresholds,
    sample based; Dice@0.5 for reference)."""

    def __init__(self):
        self.dice = DICEHandler(with_dynamic=True, with_adaptive=False, with_binary=True, sample_based=True)
        self.iou = IOUHandler(with_dynamic=True, with_adaptive=False, with_binary=True, sample_based=True)
        self.metric = FmeasureV2(metric_handlers={"dice": self.dice, "iou": self.iou})
        self.names = []

    def add(self, name, pred_u8, gt_u8):
        if pred_u8.shape != gt_u8.shape:
            pred_u8 = cv2.resize(pred_u8, (gt_u8.shape[1], gt_u8.shape[0]))
        self.metric.step(pred=pred_u8, gt=gt_u8, normalize=True)
        self.names.append(name)

    def results(self):
        r = self.metric.get_results()
        per_image = pd.DataFrame({
            "image": self.names,
            "dice_mean_thr": np.mean(self.dice.dynamic_results, axis=1),
            "iou_mean_thr": np.mean(self.iou.dynamic_results, axis=1),
            "dice_0.5": np.ravel(self.dice.binary_results),
            "iou_0.5": np.ravel(self.iou.binary_results),
        })
        summary = {
            "n": len(self.names),
            "meanDice": float(r["dice"]["dynamic"].mean()),
            "meanIoU": float(r["iou"]["dynamic"].mean()),
            "Dice@0.5": float(r["dice"]["binary"]),
            "IoU@0.5": float(r["iou"]["binary"]),
        }
        return summary, per_image


@torch.no_grad()
def evaluate(model, rows, data_dir, size, device, save_dir=None):
    """Score `model` on every row of `rows` (a slice of the split table), grouped by source dataset.
    Returns (one summary row per dataset, one row per image). With save_dir, also writes the
    prediction maps to save_dir/<dataset>/<stem>.png."""
    model.eval()
    data_dir = Path(data_dir)
    summaries, tables = [], []
    for source, group in rows.groupby("source", sort=False):
        scorer = Scorer()
        if save_dir is not None:
            (Path(save_dir) / source).mkdir(parents=True, exist_ok=True)
        for r in group.itertuples():
            gt = cv2.imread(str(data_dir / r.mask), cv2.IMREAD_GRAYSCALE)
            pred = predict_map(model, read_rgb(data_dir / r.image), size, gt.shape, device)
            if save_dir is not None:
                cv2.imwrite(str(Path(save_dir) / source / f"{r.stem}.png"), pred)
            scorer.add(f"{r.stem}.png", pred, gt)
        summary, table = scorer.results()
        summaries.append({"dataset": source, **summary})
        tables.append(table.assign(dataset=source))
    return pd.DataFrame(summaries), pd.concat(tables, ignore_index=True)


def pooled_mean_dice(per_image):
    """Mean Dice over all images together (the 256-threshold average is linear, so this equals
    the per-image average of dice_mean_thr)."""
    return float(per_image["dice_mean_thr"].mean())


# ============================================================================ training

@dataclass
class Config:
    name: str
    loss: str = "structure_official"   # see LOSSES
    augmentation: str = "none"         # see AUGMENTATIONS
    size: int = 352                    # train AND test size (a multiple of 32)
    multiscale: bool = True            # official: every batch at 0.75x, 1x and 1.25x the size
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4                   # AdamW, constant (the official schedule never decays in 100 epochs)
    weight_decay: float = 1e-4
    clip: float = 0.5                  # official gradient clamp
    seed: int = 0
    amp: bool = True                   # mixed precision on the GPU (about 2x faster on a T4)
    track_test: bool = False           # ALSO score the test sets every epoch, for analysis only
    num_workers: int = 2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scaled(x, size, mode):
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode=mode, align_corners=True)


def train(cfg, splits, data_dir, pvt_repo, out_root, device="cuda", maps_dir=None, log=print):
    """Train one configuration. Files in out_root/<name>/:
        config.json, history.csv (one row per epoch), best.pth (best validation epoch),
        last.pth (only while running, to resume), test_summary.csv / test_per_image.csv
        (chosen checkpoint on the five test sets), done.json (summary of the run).
    A finished run (done.json present) is skipped, so the notebook can simply be re-run."""
    run_dir = Path(out_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "done.json"
    if done.exists():
        log(f"[{cfg.name}] already finished - skipping (delete {done} to retrain)")
        return json.loads(done.read_text())
    if cfg.loss not in LOSSES:
        raise ValueError(f"unknown loss {cfg.loss!r}, choose from {list(LOSSES)}")
    if cfg.size % 32:
        raise ValueError("size must be a multiple of 32")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    seed_everything(cfg.seed)
    device = torch.device(device)
    data_dir = Path(data_dir)
    train_rows = splits[splits.split == "train"]
    val_rows = splits[splits.split == "val"]
    test_rows = splits[splits.split == "test"]

    loader = DataLoader(TrainSet(train_rows, data_dir, cfg.size, cfg.augmentation),
                        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0)
    model = build_model(pvt_repo).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    loss_fn = LOSSES[cfg.loss]
    sizes = sorted({int(round(cfg.size * r / 32) * 32) for r in ([0.75, 1, 1.25] if cfg.multiscale else [1])})

    history, best, start = [], -1.0, 1
    last = run_dir / "last.pth"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        history, best, start = ck["history"], ck["best"], ck["epoch"] + 1
        log(f"[{cfg.name}] resuming after epoch {ck['epoch']}")

    log(f"[{cfg.name}] {len(train_rows)} train / {len(val_rows)} val images, loss={cfg.loss}, "
        f"augmentation={cfg.augmentation}, size={cfg.size}, scales={sizes}, epochs={cfg.epochs}")
    for epoch in range(start, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            for size in sizes:                     # official multi-scale training: one step per scale
                x, y = _scaled(images, size, "bilinear"), _scaled(masks, size, "bilinear")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    p1, p2 = model(x)
                loss = loss_fn(p1.float(), y) + loss_fn(p2.float(), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                for p in model.parameters():       # official clip_gradient: clamp every gradient
                    if p.grad is not None:
                        p.grad.clamp_(-cfg.clip, cfg.clip)
                scaler.step(optimizer)
                scaler.update()
                if size == cfg.size:
                    losses.append(loss.item())
        train_seconds = time.time() - t0

        _, val_images = evaluate(model, val_rows, data_dir, cfg.size, device)
        val_dice = pooled_mean_dice(val_images)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_meanDice": val_dice,
               "val_meanIoU": float(val_images["iou_mean_thr"].mean()),
               "val_Dice@0.5": float(val_images["dice_0.5"].mean()), "train_seconds": round(train_seconds, 1)}
        if cfg.track_test:                         # monitoring only - never used to pick the model
            test_sum, _ = evaluate(model, test_rows, data_dir, cfg.size, device)
            row.update({f"test_{d}": v for d, v in zip(test_sum.dataset, test_sum.meanDice)})
        row["is_best"] = val_dice > best
        if row["is_best"]:
            best = val_dice
            torch.save(model.state_dict(), run_dir / "best.pth")
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "best": best, "epoch": epoch},
                   run_dir / "last.tmp")
        os.replace(run_dir / "last.tmp", last)     # atomic, so a disconnect never leaves half a file
        log(f"[{cfg.name}] epoch {epoch:3d}/{cfg.epochs}  loss {row['train_loss']:.4f}  "
            f"val meanDice {val_dice:.4f}{'  *best*' if row['is_best'] else ''}  ({train_seconds:.0f}s)")

    # the test sets are used here, once, with the epoch chosen on validation
    last_sum, _ = evaluate(model, test_rows, data_dir, cfg.size, device)          # last epoch, for reference
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    save_dir = Path(maps_dir) / cfg.name if maps_dir else None
    test_sum, test_images = evaluate(model, test_rows, data_dir, cfg.size, device, save_dir=save_dir)
    test_sum.to_csv(run_dir / "test_summary.csv", index=False)
    test_images.to_csv(run_dir / "test_per_image.csv", index=False)

    hist = pd.DataFrame(history)
    best_row = hist.loc[hist.val_meanDice.idxmax()]
    info = {**asdict(cfg), "best_epoch": int(best_row.epoch), "best_val_meanDice": float(best_row.val_meanDice),
            "minutes": round(float(hist.train_seconds.sum()) / 60, 1),
            **{f"test_{d}": float(v) for d, v in zip(test_sum.dataset, test_sum.meanDice)},
            **{f"testIoU_{d}": float(v) for d, v in zip(test_sum.dataset, test_sum.meanIoU)},
            **{f"lastepoch_test_{d}": float(v) for d, v in zip(last_sum.dataset, last_sum.meanDice)}}
    done.write_text(json.dumps(info, indent=2))
    last.unlink()                                  # ~300 MB with the optimiser state; best.pth stays
    log(f"[{cfg.name}] done: best epoch {info['best_epoch']} (val {info['best_val_meanDice']:.4f}); test " +
        "  ".join(f"{d} {info[f'test_{d}']:.3f}" for d in TEST_SETS))
    return info


def collect(out_root, names=None):
    """One row per finished run with its settings, chosen epoch, validation Dice and test Dice / IoU
    per dataset, plus the plain average over the five test sets."""
    rows = []
    for done in sorted(Path(out_root).glob("*/done.json")):
        info = json.loads(done.read_text())
        if names is None or info["name"] in names:
            rows.append(info)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["test_mean5"] = df[[f"test_{d}" for d in TEST_SETS]].mean(axis=1)
    df["testIoU_mean5"] = df[[f"testIoU_{d}" for d in TEST_SETS]].mean(axis=1)
    if names is not None:
        df = df.set_index("name").loc[[n for n in names if n in set(df.name)]].reset_index()
    return df


# ============================================================================ MedSAM (optional)

def boxes_from_mask(mask_u8, min_area=10, jitter=0.0, rng=None):
    """One [x0, y0, x1, y1] box per connected polyp region of a 0/255 mask. jitter > 0 moves every
    side by up to jitter x the box width / height, to imitate a box drawn by hand."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 127).astype(np.uint8), connectivity=8)
    h, w = mask_u8.shape
    rng = rng or np.random.default_rng(0)
    boxes = []
    for x, y, bw, bh, area in stats[1:]:
        if area < min_area:
            continue
        box = np.array([x, y, x + bw, y + bh], np.float32)
        if jitter:
            box += rng.uniform(-jitter, jitter, 4) * np.array([bw, bh, bw, bh])
        boxes.append(np.clip(box, 0, [w, h, w, h]))
    return np.array(boxes, np.float32).reshape(-1, 4)


def load_medsam(medsam_repo, checkpoint, device):
    medsam_repo = str(Path(medsam_repo).resolve())
    if medsam_repo not in sys.path:
        sys.path.insert(0, medsam_repo)
    from segment_anything import sam_model_registry

    return sam_model_registry["vit_b"](checkpoint=checkpoint).to(device).eval()


@torch.no_grad()
def medsam_map(model, rgb, boxes, device):
    """MedSAM probability map for all boxes of one image (union = pixel-wise max), preprocessing as
    in MedSAM_Inference.py: 1024 x 1024 cubic resize, min-max to [0, 1], no ImageNet normalisation."""
    from skimage import transform

    h, w = rgb.shape[:2]
    if len(boxes) == 0:
        return np.zeros((h, w), np.uint8)
    img = transform.resize(rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)
    img = (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None)
    x = torch.tensor(img).float().permute(2, 0, 1)[None].to(device)
    embedding = model.image_encoder(x)
    box_1024 = torch.as_tensor(boxes / np.array([w, h, w, h]) * 1024, dtype=torch.float, device=device)[:, None, :]
    sparse, dense = model.prompt_encoder(points=None, boxes=box_1024, masks=None)
    logits, _ = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                                   sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                   multimask_output=False)
    prob = F.interpolate(torch.sigmoid(logits), size=(h, w), mode="bilinear", align_corners=False)
    prob = prob.max(dim=0).values.squeeze().cpu().numpy()
    return np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)


def evaluate_medsam(model, rows, data_dir, device, box_source="gt", pred_maps=None, jitter=0.0, save_dir=None):
    """MedSAM on the test rows. box_source='gt': boxes from the ground-truth mask (an expert drawing a
    box - semi-automatic). box_source='pred': boxes from a model's prediction map in
    pred_maps/<dataset>/<stem>.png thresholded at 0.5 (a fully automatic two-stage pipeline)."""
    data_dir = Path(data_dir)
    rng = np.random.default_rng(0)
    summaries, tables = [], []
    for source, group in rows.groupby("source", sort=False):
        scorer = Scorer()
        if save_dir is not None:
            (Path(save_dir) / source).mkdir(parents=True, exist_ok=True)
        for r in group.itertuples():
            gt = cv2.imread(str(data_dir / r.mask), cv2.IMREAD_GRAYSCALE)
            if box_source == "gt":
                box_mask = gt
            else:
                box_mask = cv2.imread(str(Path(pred_maps) / source / f"{r.stem}.png"), cv2.IMREAD_GRAYSCALE)
            boxes = boxes_from_mask(box_mask, jitter=jitter, rng=rng)
            pred = medsam_map(model, read_rgb(data_dir / r.image), boxes, device)
            if save_dir is not None:
                cv2.imwrite(str(Path(save_dir) / source / f"{r.stem}.png"), pred)
            scorer.add(f"{r.stem}.png", pred, gt)
        summary, table = scorer.results()
        summaries.append({"dataset": source, **summary})
        tables.append(table.assign(dataset=source))
    return pd.DataFrame(summaries), pd.concat(tables, ignore_index=True)

"""Leakage-controlled Milestone 2 training utilities for official Polyp-PVT.

This module is designed for Google Colab. It never reads an official test set while
training or validating. ClinicDB is retained wholly in training when sequence IDs
are unavailable; validation is sampled from Kvasir-SEG only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from py_sod_metrics import DICEHandler, FmeasureV2, IOUHandler
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_COUNTS = {"Kvasir": 900, "CVC-ClinicDB": 550}
TEST_COUNTS = {
    "Kvasir": 100,
    "CVC-ClinicDB": 62,
    "CVC-ColonDB": 380,
    "ETIS-LaribPolypDB": 196,
    "CVC-300": 60,
}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _index_by_stem(folder: Path, extensions: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in Path(folder).iterdir():
        if path.name.startswith(".") or path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            raise ValueError(f"Multiple files with stem {path.stem!r} in {folder}")
        result[path.stem] = path
    return result


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dhash64(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).ravel():
        value = (value << 1) | int(bit)
    return value


def make_split(data_root: str | Path, out_csv: str | Path, seed: int = 2026,
               val_fraction: float = 0.10, dhash_distance: int = 3) -> pd.DataFrame:
    """Pair and audit official TrainDataset, then hold out Kvasir groups only.

    ClinicDB remains wholly in training because the archive has numbered frames
    but no verified video sequence manifest. Exact/near duplicate components are
    never split across train and validation.
    """
    root = Path(data_root)
    images = _index_by_stem(root / "TrainDataset" / "image", IMAGE_EXTENSIONS)
    masks = _index_by_stem(root / "TrainDataset" / "masks", MASK_EXTENSIONS)
    if set(images) != set(masks):
        raise ValueError(
            f"Pair mismatch: image-only={sorted(set(images)-set(masks))[:10]}, "
            f"mask-only={sorted(set(masks)-set(images))[:10]}"
        )

    records = []
    for stem in sorted(images):
        source = "CVC-ClinicDB" if stem.isdigit() else "Kvasir"
        image = cv2.imread(str(images[stem]))
        mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unreadable image/mask pair: {stem}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask dimensions differ for {stem}")
        records.append({
            "stem": stem,
            "source": source,
            "image": str(images[stem].relative_to(root)),
            "mask": str(masks[stem].relative_to(root)),
            "image_sha256": _hash_file(images[stem]),
            "dhash": _dhash64(images[stem]),
        })
    frame = pd.DataFrame(records)
    counts = frame.groupby("source").size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected official training-pool counts: {counts}")

    # Connected components over the training pool so near-duplicates cannot
    # cross the partition. Any component containing ClinicDB stays in train.
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = frame["dhash"].tolist()
    for i, value in enumerate(hashes):
        for j in range(i):
            if (value ^ hashes[j]).bit_count() <= dhash_distance:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(frame)):
        groups.setdefault(find(index), []).append(index)
    candidates = [
        indices for indices in groups.values()
        if all(frame.iloc[i]["source"] == "Kvasir" for i in indices)
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    target = round(val_fraction * EXPECTED_COUNTS["Kvasir"])
    validation: set[int] = set()
    held = 0
    for indices in candidates:
        if held >= target:
            break
        validation.update(indices)
        held += len(indices)
    if held < max(1, target - 20):
        raise RuntimeError(f"Near-duplicate groups left too few validation images: {held}")

    frame["split"] = ["val" if i in validation else "train" for i in range(len(frame))]
    frame["seed"] = seed
    frame["dhash_group"] = [find(i) for i in range(len(frame))]
    if (frame["source"].eq("CVC-ClinicDB") & frame["split"].eq("val")).any():
        raise AssertionError("ClinicDB must remain wholly in training")
    val_groups = set(frame.loc[frame["split"] == "val", "dhash_group"])
    train_groups = set(frame.loc[frame["split"] == "train", "dhash_group"])
    if val_groups & train_groups:
        raise AssertionError("Near-duplicate group leakage detected")
    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


class PolypDataset(Dataset):
    """Square-resized RGB inputs and binary masks, optionally paired-augmented."""

    def __init__(self, rows: pd.DataFrame, data_root: str | Path, size: int,
                 augment: bool = False, include_native_mask: bool = False):
        if size % 32:
            raise ValueError(f"Input size {size} must be divisible by 32")
        self.rows = rows.reset_index(drop=True)
        self.root = Path(data_root)
        self.size = size
        self.augment = augment
        self.include_native_mask = include_native_mask

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image_path, mask_path = self.root / row.image, self.root / row["mask"]
        image = cv2.imread(str(image_path))
        mask_native = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask_native is None:
            raise FileNotFoundError(f"Could not read pair: {image_path}, {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask_native > 127).astype(np.uint8)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            k = random.randrange(4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
        x = np.ascontiguousarray(image).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = np.ascontiguousarray(mask[None]).astype(np.float32)
        result = (torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y), row.source)
        if self.include_native_mask:
            result += (torch.from_numpy((mask_native > 127).astype(np.uint8)),)
        return result


def structure_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = 1 + 5 * torch.abs(F.avg_pool2d(target, 31, 1, 15) - target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (weight * bce).sum((2, 3)) / weight.sum((2, 3))
    prob = torch.sigmoid(logits)
    inter = (prob * target * weight).sum((2, 3))
    union = ((prob + target) * weight).sum((2, 3))
    weighted_iou = 1 - (inter + 1) / (union - inter + 1)
    return (weighted_bce + weighted_iou).mean()


def bce_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dice = (2 * (prob * target).sum((1, 2, 3)) + eps) / (
        prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    )
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()


def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 0.3, beta: float = 0.7,
                       gamma: float = 4 / 3, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum((1, 2, 3))
    fp = (prob * (1 - target)).sum((1, 2, 3))
    fn = ((1 - prob) * target).sum((1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - tversky).clamp_min(0).pow(gamma).mean()


def combined_loss(outputs, target: torch.Tensor, name: str) -> torch.Tensor:
    functions = {
        "structure": structure_loss,
        "bce_dice": bce_soft_dice_loss,
        "focal_tversky": focal_tversky_loss,
    }
    if name not in functions:
        raise ValueError(f"Unknown loss: {name}")
    predictions = list(outputs)
    fn = functions[name]
    losses = []
    for prediction in predictions:
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(prediction, size=target.shape[-2:],
                                       mode="bilinear", align_corners=False)
        losses.append(fn(prediction, target))
    return sum(losses)


def _new_metric():
    return FmeasureV2(metric_handlers={
        "dice": DICEHandler(with_dynamic=True, with_adaptive=False,
                            with_binary=False, sample_based=True),
        "iou": IOUHandler(with_dynamic=True, with_adaptive=False,
                          with_binary=False, sample_based=True),
    })


@torch.no_grad()
def validate(model: torch.nn.Module, rows: pd.DataFrame, data_root: str | Path,
             size: int, device: torch.device) -> tuple[float, float]:
    """Dynamic 256-threshold mean Dice/IoU, with predictions restored to native size."""
    model.eval()
    loader = DataLoader(
        PolypDataset(rows, data_root, size, include_native_mask=True),
        batch_size=1, shuffle=False, num_workers=0,
    )
    metric = _new_metric()
    for image, _, _, native_mask in loader:
        first, second = model(image.to(device))
        prob = torch.sigmoid(F.interpolate(
            first + second, size=native_mask.shape[-2:],
            mode="bilinear", align_corners=False
        ))[0, 0].float().cpu().numpy()
        pred = np.clip(prob * 255, 0, 255).astype(np.uint8)
        gt = (native_mask[0].cpu().numpy().astype(np.uint8) * 255)
        metric.step(pred=pred, gt=gt, normalize=True)
    result = metric.get_results()
    return (
        float(result["dice"]["dynamic"].mean()),
        float(result["iou"]["dynamic"].mean()),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _atomic_save(value, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def train_experiment(
    config: dict,
    model_factory: Callable[[], torch.nn.Module],
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    data_root: str | Path,
    run_root: str | Path,
    runtime_versions: dict,
    seed: int = 2026,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 0.5,
    amp: bool = True,
    resume: bool = True,
) -> dict:
    """Train one run, selecting checkpoints from validation only; save resumably."""
    size = int(config["size"])
    if size not in (288, 352, 416) or size % 32:
        raise ValueError("Supported input sizes are 288, 352, and 416 (multiples of 32).")
    device = torch.device("cuda")
    run_dir = Path(run_root) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        **config, "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "grad_clip": grad_clip, "amp": amp, "runtime_versions": runtime_versions,
    }
    (run_dir / "config.json").write_text(json.dumps(full_config, indent=2))
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PolypDataset(train_rows, data_root, size, augment=bool(config["augmentation"])),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
        generator=generator,
    )
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    start_epoch, best_pair, elapsed = 0, (-1.0, -1.0), 0.0
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Resume config differs from saved checkpoint.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_pair, elapsed = tuple(state["best_pair"]), state["elapsed_seconds"]
        py, np_state, cpu_state, cuda_state, loader_state = state["rng"]
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)
        generator.set_state(loader_state)

    log_path = run_dir / "epochs.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as stream:
            csv.writer(stream).writerow([
                "epoch", "train_loss", "val_mean_dice", "val_mean_iou",
                "epoch_seconds", "elapsed_seconds",
            ])

    for epoch in range(start_epoch, epochs):
        started = time.time()
        model.train()
        losses = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outputs = model(images)
                loss = combined_loss(outputs, masks, config["loss"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_dice, val_iou = validate(model, val_rows, data_root, size, device)
        epoch_seconds = time.time() - started
        elapsed += epoch_seconds
        if (val_dice, val_iou) > best_pair:
            best_pair = (val_dice, val_iou)
            _atomic_save({
                "model": model.state_dict(), "config": config, "epoch": epoch + 1,
                "val_mean_dice": val_dice, "val_mean_iou": val_iou,
            }, best_path)

        with open(log_path, "a", newline="") as stream:
            csv.writer(stream).writerow([
                epoch + 1, float(np.mean(losses)), val_dice, val_iou,
                epoch_seconds, elapsed,
            ])
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch,
            "best_pair": best_pair, "elapsed_seconds": elapsed, "config": config,
            "rng": (
                random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all(), generator.get_state(),
            ),
        }
        _atomic_save(state, last_path)
        print(
            f'{config["name"]}: epoch {epoch + 1}/{epochs}, '
            f'loss={np.mean(losses):.4f}, val Dice={val_dice:.4f}, '
            f'val IoU={val_iou:.4f}, {epoch_seconds / 60:.1f} min'
        )
    return {
        "experiment": config["name"], "size": size, "loss": config["loss"],
        "augmentation": config["augmentation"],
        "best_val_mean_dice": best_pair[0], "best_val_mean_iou": best_pair[1],
        "training_seconds": elapsed, "status": "complete",
    }


def summarize_runs(experiments: list[dict], run_root: str | Path,
                   epochs: int = 100) -> pd.DataFrame:
    rows = []
    for config in experiments:
        path = Path(run_root) / config["name"] / "epochs.csv"
        if not path.exists():
            continue
        history = pd.read_csv(path)
        if len(history) < epochs:
            continue
        best = history.sort_values(
            ["val_mean_dice", "val_mean_iou"], ascending=False
        ).iloc[0]
        rows.append({
            "experiment": config["name"], "size": config["size"],
            "loss": config["loss"], "augmentation": config["augmentation"],
            "best_epoch": int(best.epoch),
            "val_mean_dice": float(best.val_mean_dice),
            "val_mean_iou": float(best.val_mean_iou),
            "training_seconds": float(history.epoch_seconds.sum()),
            "status": "complete",
        })
    return pd.DataFrame(rows)


def lock_validation_selection(summary: pd.DataFrame, experiments: list[dict],
                              run_root: str | Path, lock_path: str | Path) -> dict:
    """Write one immutable final selection record, based only on validation."""
    path = Path(lock_path)
    if path.exists():
        raise FileExistsError(f"Selection already locked: {path}")
    if len(summary) != len(experiments):
        raise RuntimeError(
            f"All {len(experiments)} runs must finish before selection; "
            f"{len(summary)} complete."
        )
    ranked = summary.sort_values(
        ["val_mean_dice", "val_mean_iou"], ascending=False
    )
    winner = ranked.iloc[0]
    record = {
        "selected_experiment": str(winner.experiment),
        "selected_checkpoint": str(Path(run_root) / str(winner.experiment) / "best.pt"),
        "selection_rule": "highest validation dynamic mean Dice; validation dynamic mean IoU tie-break",
        "validation_mean_dice": float(winner.val_mean_dice),
        "validation_mean_iou": float(winner.val_mean_iou),
        "locked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "test_evaluated": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return record


def audit_test_duplicates(manifest: pd.DataFrame, test_root: str | Path,
                          distance: int = 3) -> list[dict]:
    """Screen locked test images against training/validation after selection."""
    train_sha = set(manifest["image_sha256"].astype(str))
    train_hashes = [int(value) for value in manifest["dhash"]]
    matches = []
    for dataset in TEST_COUNTS:
        images = _index_by_stem(Path(test_root) / dataset / "images", IMAGE_EXTENSIONS)
        for stem, path in images.items():
            exact = _hash_file(path) in train_sha
            value = _dhash64(path)
            nearest = min(((value ^ other).bit_count() for other in train_hashes), default=64)
            if exact or nearest <= distance:
                matches.append({"dataset": dataset, "stem": stem,
                    "exact_sha256_match": exact, "nearest_dhash_distance": nearest})
    return matches

@torch.no_grad()
def evaluate_locked_test(
    model_factory: Callable[[], torch.nn.Module],
    data_root: str | Path,
    lock_path: str | Path,
    results_path: str | Path,
    manifest: pd.DataFrame | None = None,
    overlap_path: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate the one validation-locked model once on all five official test sets."""
    lock_path, results_path = Path(lock_path), Path(results_path)
    if not lock_path.exists():
        raise RuntimeError("Create the validation selection lock before test evaluation.")
    lock = json.loads(lock_path.read_text())
    if lock.get("test_evaluated") or results_path.exists():
        raise RuntimeError("Final test evaluation is already recorded; refusing a second run.")
    if manifest is not None:
        overlaps = audit_test_duplicates(manifest, Path(data_root) / "TestDataset")
        if overlap_path is not None:
            Path(overlap_path).write_text(json.dumps({"checked_after_selection": True, "matches": overlaps}, indent=2))
        if overlaps:
            raise RuntimeError(f"Potential train/test image overlap; no test metrics computed. See {overlap_path}")
    selected = lock["selected_experiment"]
    checkpoint = torch.load(lock["selected_checkpoint"], map_location="cpu",
                            weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda")
    model = model_factory().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows, root = [], Path(data_root)
    for dataset, expected in TEST_COUNTS.items():
        image_dir = root / "TestDataset" / dataset / "images"
        mask_dir = root / "TestDataset" / dataset / "masks"
        images = _index_by_stem(image_dir, IMAGE_EXTENSIONS)
        masks = _index_by_stem(mask_dir, MASK_EXTENSIONS)
        if set(images) != set(masks) or len(images) != expected:
            raise ValueError(
                f"Invalid locked test set {dataset}: images={len(images)}, "
                f"masks={len(masks)}, expected={expected}"
            )
        metric = _new_metric()
        for stem in sorted(images):
            image = cv2.imread(str(images[stem]))
            gt = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
            if image is None or gt is None:
                raise ValueError(f"Unreadable test pair {dataset}/{stem}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(config["size"]), int(config["size"])),
                               interpolation=cv2.INTER_LINEAR)
            x = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(device)
            first, second = model(tensor)
            prob = torch.sigmoid(F.interpolate(
                first + second, size=gt.shape, mode="bilinear", align_corners=False
            ))[0, 0].cpu().numpy()
            metric.step(
                pred=np.clip(prob * 255, 0, 255).astype(np.uint8),
                gt=((gt > 127) * 255).astype(np.uint8),
                normalize=True,
            )
        result = metric.get_results()
        rows.append({
            "model": selected, "dataset": dataset, "n": expected,
            "input_size": int(config["size"]),
            "mean_dice": float(result["dice"]["dynamic"].mean()),
            "mean_iou": float(result["iou"]["dynamic"].mean()),
        })

    frame = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = results_path.with_name(results_path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, results_path)
    lock["test_evaluated"] = True
    lock["test_results_file"] = str(results_path)
    lock["test_evaluated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    lock_path.write_text(json.dumps(lock, indent=2))
    return frame

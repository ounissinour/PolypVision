"""Data exploration and preparation for training: train / validation / test split and PyTorch loaders.

The official repositories train on TrainDataset (Kvasir-SEG 900 + CVC-ClinicDB 550 images) and
report results on the five test sets, with no validation set. For our own training in Milestone 2
we need one, so this script

  1. lists every image/mask pair of TrainDataset and the five test sets,
  2. holds out a validation part of TrainDataset (10 % by default). Kvasir images are independent,
     so they are sampled at random. CVC-ClinicDB frames come from videos and consecutive frame
     numbers belong to the same video, so whole blocks of consecutive frames are held out, never
     single frames (a frame next to a training frame would be an almost-duplicate),
  3. explores the data (counts, image sizes, polyp size),
  4. builds PyTorch datasets that return (image, mask) tensors at the training size used by both
     papers (352 x 352, ImageNet normalisation, 0/1 masks).

Usage after the download cell of M1_Reproduce.ipynb (or scripts of the same layout):
  python prepare_data.py --data /content/data --out /content/m1
Writes <out>/results/splits.csv and <out>/figures/data_*.png, prints the summary tables and
the shape of one batch per split.
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

TEST_SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
TRAIN_SIZE = 352                    # both papers train and test at 352 x 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLINICDB_BLOCK = 25                 # consecutive CVC-ClinicDB frames kept together when splitting


# ----------------------------------------------------------------------------- listing and split

def source_of(stem):
    """CVC-ClinicDB frames are numbered (1.png ... 612.png); Kvasir-SEG names are alphanumeric."""
    return "CVC-ClinicDB" if stem.isdigit() else "Kvasir"


def list_pairs(data_dir, image_dir, mask_dir):
    """Image/mask pairs matched by file stem, with paths relative to data_dir, as a DataFrame."""
    masks = {p.stem: p for p in Path(mask_dir).iterdir() if not p.name.startswith(".")}
    rows = []
    for img in sorted(Path(image_dir).iterdir()):
        if img.name.startswith(".") or img.stem not in masks:
            continue
        rows.append({"image": str(img.relative_to(data_dir)), "mask": str(masks[img.stem].relative_to(data_dir)),
                     "stem": img.stem})
    if not rows:
        raise FileNotFoundError(f"no image/mask pairs in {image_dir}")
    return pd.DataFrame(rows)


def make_split(data_dir, val_fraction=0.1, seed=0):
    """One row per image: split (train / val / test), source dataset, and image / mask paths
    relative to data_dir."""
    data_dir = Path(data_dir)
    train = list_pairs(data_dir, data_dir / "TrainDataset" / "image", data_dir / "TrainDataset" / "masks")
    train["source"] = train["stem"].map(source_of)
    train["split"] = "train"
    rng = random.Random(seed)

    # Kvasir: independent images, plain random sample
    kvasir = train.index[train.source == "Kvasir"].tolist()
    for i in rng.sample(kvasir, round(val_fraction * len(kvasir))):
        train.loc[i, "split"] = "val"

    # CVC-ClinicDB: blocks of consecutive frame numbers, so a video is never split across train and val
    clinic = train[train.source == "CVC-ClinicDB"].copy()
    clinic["frame"] = clinic["stem"].astype(int)
    clinic = clinic.sort_values("frame")
    blocks = [clinic.index[i:i + CLINICDB_BLOCK].tolist() for i in range(0, len(clinic), CLINICDB_BLOCK)]
    rng.shuffle(blocks)
    held = 0
    for block in blocks:
        if held >= val_fraction * len(clinic):
            break
        train.loc[block, "split"] = "val"
        held += len(block)

    tests = []
    for name in TEST_SETS:
        t = list_pairs(data_dir, data_dir / "TestDataset" / name / "images", data_dir / "TestDataset" / name / "masks")
        t["source"], t["split"] = name, "test"
        tests.append(t)
    return pd.concat([train] + tests, ignore_index=True)[["split", "source", "stem", "image", "mask"]]


# ----------------------------------------------------------------------------- exploration

def explore(df, data_dir):
    """Counts per split and source, image sizes, and polyp size as a fraction of the image."""
    sizes, areas = [], []
    for row in df.itertuples():
        mask = cv2.imread(str(Path(data_dir) / row.mask), cv2.IMREAD_GRAYSCALE)
        sizes.append(f"{mask.shape[1]}x{mask.shape[0]}")
        areas.append(float((mask > 127).mean()))
    df = df.assign(size=sizes, polyp_fraction=areas)
    counts = df.pivot_table(index="source", columns="split", values="stem", aggfunc="count", fill_value=0)
    stats = (df.groupby(["split", "source"])
               .agg(images=("stem", "count"),
                    sizes=("size", "nunique"),
                    most_common_size=("size", lambda s: s.mode().iloc[0]),
                    polyp_fraction_median=("polyp_fraction", "median"),
                    polyp_fraction_min=("polyp_fraction", "min"),
                    polyp_fraction_max=("polyp_fraction", "max"))
               .round(3))
    return df, counts, stats


def plot_samples(df, data_dir, out_png, per_group=3, seed=0):
    """A few images with their mask outline from each (split, source) group."""
    groups = [g for g in df.groupby(["split", "source"], sort=False)]
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(groups), per_group, figsize=(3 * per_group, 2.6 * len(groups)))
    for ax_row, ((split, source), g) in zip(axes, groups):
        for ax, row in zip(ax_row, g.sample(min(per_group, len(g)), random_state=rng.randint(0, 10**6)).itertuples()):
            image = cv2.imread(str(Path(data_dir) / row.image))[:, :, ::-1]
            mask = cv2.imread(str(Path(data_dir) / row.mask), cv2.IMREAD_GRAYSCALE)
            ax.imshow(image)
            ax.contour(mask > 127, levels=[0.5], colors="yellow", linewidths=1)
            ax.set_title(f"{split} / {source}\n{row.stem[:18]}  {image.shape[1]}x{image.shape[0]}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    return fig


def plot_stats(df, out_png):
    """Polyp size distribution per source, and image count per split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for source, g in df.groupby("source"):
        axes[0].hist(g["polyp_fraction"], bins=30, range=(0, 0.6), alpha=0.5, label=f"{source} ({len(g)})")
    axes[0].set_xlabel("polyp area / image area")
    axes[0].set_ylabel("images")
    axes[0].set_title("polyp size")
    axes[0].legend(fontsize=8)
    counts = df.groupby(["split", "source"]).size().unstack(fill_value=0).loc[["train", "val", "test"]]
    counts.plot.bar(ax=axes[1], stacked=True)
    axes[1].set_title("images per split")
    axes[1].set_ylabel("images")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    return fig


# ----------------------------------------------------------------------------- PyTorch datasets

class PolypDataset(Dataset):
    """(image, mask) pairs resized to size x size. Image: float tensor 3xHxW, ImageNet-normalised.
    Mask: float tensor 1xHxW with values 0/1. augment=True adds random flips for training."""

    def __init__(self, rows, data_dir, size=TRAIN_SIZE, augment=False):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.size = size
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        image = cv2.imread(str(self.data_dir / row.image))[:, :, ::-1]          # BGR -> RGB
        mask = cv2.imread(str(self.data_dir / row["mask"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        image = (image - self.mean) / self.std
        mask = torch.from_numpy(np.ascontiguousarray(mask > 127)).float().unsqueeze(0)
        return image, mask


def make_loaders(df, data_dir, size=TRAIN_SIZE, batch_size=16, num_workers=2):
    """Train (shuffled, augmented), validation and test loaders from the split table."""
    return {
        "train": DataLoader(PolypDataset(df[df.split == "train"], data_dir, size, augment=True),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "val": DataLoader(PolypDataset(df[df.split == "val"], data_dir, size),
                          batch_size=batch_size, num_workers=num_workers),
        "test": DataLoader(PolypDataset(df[df.split == "test"], data_dir, size),
                           batch_size=batch_size, num_workers=num_workers),
    }


# ----------------------------------------------------------------------------- script

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder holding TrainDataset/ and TestDataset/")
    parser.add_argument("--out", default=".", help="where results/splits.csv and figures/ go")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = make_split(args.data, args.val_fraction, args.seed)
    df, counts, stats = explore(df, args.data)
    df.to_csv(out / "results" / "splits.csv", index=False)
    print("images per source and split\n", counts, "\n")
    print(stats.to_string(), "\n")
    plot_samples(df, args.data, out / "figures" / "data_samples.png")
    plot_stats(df, out / "figures" / "data_stats.png")

    loaders = make_loaders(df, args.data, num_workers=0)
    for split, loader in loaders.items():
        images, masks = next(iter(loader))
        print(f"{split:5s} {len(loader.dataset):4d} images  batch: inputs {tuple(images.shape)} "
              f"outputs {tuple(masks.shape)}  mask values {sorted(masks.unique().tolist())}")


if __name__ == "__main__":
    main()

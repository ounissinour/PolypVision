
"""
metrics.py - Shared evaluation utilities for PolypVision.

Computes per-image Dice and IoU between predicted and ground-truth masks,
and aggregates them per dataset. Used for all models (PraNet, Polyp-PVT, ...)
so that results are directly comparable.

Evaluation choices (document these in the report):
  - Predictions are grayscale maps in [0, 255], same size as the ground truth.
  - Binarization: prediction >= 0.5 (i.e. >= 127.5 on the 0-255 scale).
  - Ground truth is binarized at 0.5 as well.
  - Dice and IoU are computed per image, then averaged over the dataset.
  - eps avoids division by zero when both masks are empty.
"""

import os
import cv2
import numpy as np
import pandas as pd


def dice_iou(pred, gt, thr=0.5, eps=1e-8):
    """Return (dice, iou) for one prediction/ground-truth pair (uint8 arrays 0-255)."""
    p = (pred / 255.0) >= thr
    g = (gt / 255.0) >= 0.5
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    dice = (2 * inter + eps) / (p.sum() + g.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


def evaluate(model_name, dataset, pred_root, gt_root, thr=0.5):
    """Evaluate one model on one dataset. Returns a DataFrame with one row per image.

    Columns: model, dataset, image, dice, iou  (the shared result format).
    """
    rows = []
    for name in sorted(os.listdir(pred_root)):
        pr = cv2.imread(os.path.join(pred_root, name), 0)
        gt = cv2.imread(os.path.join(gt_root, name), 0)
        if pr is None or gt is None:
            print(f"Skipping {name}: could not read prediction or ground truth")
            continue
        if pr.shape != gt.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]))
        d, i = dice_iou(pr, gt, thr=thr)
        rows.append(dict(model=model_name, dataset=dataset, image=name, dice=d, iou=i))
    return pd.DataFrame(rows)


def summarize(df):
    """Mean Dice and IoU per model and dataset."""
    return df.groupby(["model", "dataset"])[["dice", "iou"]].mean().round(4)

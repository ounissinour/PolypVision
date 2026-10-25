"""
metrics.py - Shared evaluation code for PolypVision.

Computes mean Dice and mean IoU with the `pysodmetrics` package, which follows
the MATLAB evaluation toolbox used in the PraNet and Polyp-PVT papers: for each
image the score is computed at 256 thresholds, averaged over images, and then
averaged over thresholds.

Expected layout:
    pred_root/<dataset>/<image>.png          prediction maps (0-255 grayscale)
    gt_root/<dataset>/masks/<image>.png      ground-truth masks

Usage:
    from metrics import evaluate_all
    df = evaluate_all("PraNet", pred_root, gt_root)
"""

import os

import cv2
import numpy as np
import pandas as pd
from py_sod_metrics import DICEHandler, FmeasureV2, IOUHandler

DATASETS = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]


def _list_images(folder):
    return sorted(n for n in os.listdir(folder) if not n.startswith("."))


def evaluate_dataset(pred_dir, gt_dir):
    """Return (n_images, mean_dice, mean_iou) for one dataset."""
    fm = FmeasureV2(
        metric_handlers={
            "dice": DICEHandler(with_dynamic=True, with_adaptive=False,
                                with_binary=False, sample_based=True),
            "iou": IOUHandler(with_dynamic=True, with_adaptive=False,
                              with_binary=False, sample_based=True),
        }
    )
    names = _list_images(pred_dir)
    for name in names:
        pred = cv2.imread(os.path.join(pred_dir, name), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            raise FileNotFoundError(f"Cannot read prediction or mask for {name}")
        gt = ((gt > 128) * 255).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        fm.step(pred=pred, gt=gt)
    res = fm.get_results()
    return len(names), float(res["dice"]["dynamic"].mean()), float(res["iou"]["dynamic"].mean())


def evaluate_all(model_name, pred_root, gt_root, datasets=DATASETS):
    """Evaluate one model on several datasets.

    Returns a DataFrame with columns: model, dataset, n_images, mean_dice, mean_iou.
    """
    rows = []
    for ds in datasets:
        n, dice, iou = evaluate_dataset(os.path.join(pred_root, ds),
                                        os.path.join(gt_root, ds, "masks"))
        rows.append(dict(model=model_name, dataset=ds, n_images=n,
                         mean_dice=round(dice, 4), mean_iou=round(iou, 4)))
    return pd.DataFrame(rows)


def compare_with_paper(df, paper):
    """Join results with paper values and add the differences.

    `paper` maps dataset -> (dice, iou), e.g. {"Kvasir": (0.898, 0.840)}.
    """
    p = pd.DataFrame(paper, index=["dice_paper", "iou_paper"]).T
    out = df.set_index("dataset").join(p)
    out["dice_diff"] = (out["mean_dice"] - out["dice_paper"]).round(4)
    out["iou_diff"] = (out["mean_iou"] - out["iou_paper"]).round(4)
    return out

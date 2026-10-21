"""Mean Dice and mean IoU with pysodmetrics, matching the MATLAB toolbox used in both papers.

The papers' "meanDic" is NOT Dice at a single threshold. The MATLAB code
(PraNet/eval/main.m) binarises each prediction at 256 thresholds (1:-1/255:0),
computes Dice per image and threshold, averages over images and then over
thresholds. pysodmetrics reproduces this with FmeasureV2 + DICEHandler/IOUHandler
in "dynamic" mode: the mean of the 256-value curve is the paper's meanDic.

For reference we also report Dice/IoU after a single binarisation at 0.5, which is
what a hand-written metric usually does; the two can differ by about 0.01.

Usage:
  python src/evaluate.py --model PraNet --pred_root <maps>/PraNet --gt_root <data>/TestDataset \
      --out results/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import py_sod_metrics as sod

DATASETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]


def evaluate_dataset(pred_dir, gt_dir):
    """Return (summary dict, per-image DataFrame) for one dataset."""
    metric = sod.FmeasureV2(metric_handlers={
        "dice": sod.DICEHandler(with_dynamic=True, with_adaptive=False, with_binary=True, sample_based=True),
        "iou": sod.IOUHandler(with_dynamic=True, with_adaptive=False, with_binary=True, sample_based=True),
    })
    gts = {p.stem: p for p in Path(gt_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".tif", ".bmp"}}
    preds = sorted(p for p in Path(pred_dir).iterdir() if p.suffix.lower() == ".png")
    if not gts:
        raise FileNotFoundError(f"{gt_dir}: no ground-truth masks")
    missing = sorted(set(gts) - {p.stem for p in preds})
    if missing:
        raise FileNotFoundError(f"{pred_dir}: {len(missing)} predictions missing, e.g. {missing[:3]}")
    extra = sorted({p.stem for p in preds} - set(gts))
    if extra:
        raise FileNotFoundError(f"{gt_dir}: {len(extra)} masks missing, e.g. {extra[:3]}")

    names = []
    for p in preds:
        pred = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gts[p.stem]), cv2.IMREAD_GRAYSCALE)
        if pred.shape != gt.shape:
            # The test scripts already upsample to GT size; this only guards odd files.
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        metric.step(pred=pred, gt=gt, normalize=True)
        names.append(p.name)

    res = metric.get_results()
    dice_h, iou_h = metric._metric_handlers["dice"], metric._metric_handlers["iou"]
    per_image = pd.DataFrame({
        "image": names,
        "dice_mean_thr": np.mean(np.asarray(dice_h.dynamic_results), axis=1),
        "iou_mean_thr": np.mean(np.asarray(iou_h.dynamic_results), axis=1),
        "dice_0.5": np.asarray(dice_h.binary_results).ravel(),
        "iou_0.5": np.asarray(iou_h.binary_results).ravel(),
    })
    summary = {
        "n": len(names),
        "meanDice": float(res["dice"]["dynamic"].mean()),
        "meanIoU": float(res["iou"]["dynamic"].mean()),
        "maxDice": float(res["dice"]["dynamic"].max()),
        "Dice@0.5": float(res["dice"]["binary"]),
        "IoU@0.5": float(res["iou"]["binary"]),
    }
    return summary, per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pred_root", required=True, help="folder with one sub-folder of maps per dataset")
    ap.add_argument("--gt_root", required=True, help="TestDataset folder (<name>/masks)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows, per_image = [], []
    for ds in args.datasets:
        summary, df = evaluate_dataset(os.path.join(args.pred_root, ds), os.path.join(args.gt_root, ds, "masks"))
        rows.append({"model": args.model, "dataset": ds, **summary})
        per_image.append(df.assign(model=args.model, dataset=ds))
        print(f"{args.model:10s} {ds:18s} n={summary['n']:4d} "
              f"meanDice={summary['meanDice']:.3f} meanIoU={summary['meanIoU']:.3f} "
              f"(Dice@0.5={summary['Dice@0.5']:.3f})")

    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{args.model}_summary.csv"), index=False)
    pd.concat(per_image).to_csv(os.path.join(args.out, f"{args.model}_per_image.csv"), index=False)


if __name__ == "__main__":
    main()

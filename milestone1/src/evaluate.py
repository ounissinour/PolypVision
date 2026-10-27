"""Mean Dice and mean IoU of prediction maps, computed the way the PraNet and Polyp-PVT papers do.

The papers' "mean Dice" is not Dice at one threshold. Their MATLAB toolbox binarises each
prediction map at 256 thresholds, computes Dice per image and threshold, then averages over
images and over thresholds. pysodmetrics (FmeasureV2, "dynamic" results) does exactly that.
Dice and IoU after a single 0.5 threshold are saved too, for reference only.

Usage:
  python src/evaluate.py --model PraNet --pred_root maps/PraNet --gt_root data/TestDataset --out results

Writes <out>/<model>_summary.csv (one row per dataset) and <out>/<model>_per_image.csv.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from py_sod_metrics import DICEHandler, FmeasureV2, IOUHandler

DATASETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
MASK_SUFFIXES = {".png", ".jpg", ".tif", ".bmp"}


def read_gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def evaluate_dataset(pred_dir, gt_dir):
    """Score every mask in gt_dir against the prediction map of the same name in pred_dir.

    Returns (summary dict for the dataset, DataFrame with one row per image).
    """
    gt_paths = sorted(p for p in Path(gt_dir).iterdir() if p.suffix.lower() in MASK_SUFFIXES)
    if not gt_paths:
        raise FileNotFoundError(f"no masks in {gt_dir}")

    dice = DICEHandler(with_dynamic=True, with_adaptive=False, with_binary=True, sample_based=True)
    iou = IOUHandler(with_dynamic=True, with_adaptive=False, with_binary=True, sample_based=True)
    metric = FmeasureV2(metric_handlers={"dice": dice, "iou": iou})

    names = []
    for gt_path in gt_paths:
        pred_path = Path(pred_dir) / (gt_path.stem + ".png")
        if not pred_path.exists():
            raise FileNotFoundError(f"missing prediction {pred_path}")
        pred, gt = read_gray(pred_path), read_gray(gt_path)
        if pred.shape != gt.shape:  # the test scripts already resize to the mask size; just in case
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        metric.step(pred=pred, gt=gt, normalize=True)
        names.append(pred_path.name)

    results = metric.get_results()
    per_image = pd.DataFrame({
        "image": names,
        "dice_mean_thr": np.mean(dice.dynamic_results, axis=1),  # mean over the 256 thresholds
        "iou_mean_thr": np.mean(iou.dynamic_results, axis=1),
        "dice_0.5": np.ravel(dice.binary_results),
        "iou_0.5": np.ravel(iou.binary_results),
    })
    summary = {
        "n": len(names),
        "meanDice": float(results["dice"]["dynamic"].mean()),
        "meanIoU": float(results["iou"]["dynamic"].mean()),
        "maxDice": float(results["dice"]["dynamic"].max()),
        "Dice@0.5": float(results["dice"]["binary"]),
        "IoU@0.5": float(results["iou"]["binary"]),
    }
    return summary, per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="name used in the output files, e.g. PraNet")
    parser.add_argument("--pred_root", required=True, help="folder with one sub-folder of maps per dataset")
    parser.add_argument("--gt_root", required=True, help="TestDataset folder (<dataset>/masks)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, per_image = [], []
    for dataset in args.datasets:
        summary, table = evaluate_dataset(Path(args.pred_root) / dataset, Path(args.gt_root) / dataset / "masks")
        summaries.append({"model": args.model, "dataset": dataset, **summary})
        per_image.append(table.assign(model=args.model, dataset=dataset))
        print(f"{args.model:10s} {dataset:18s} n={summary['n']:4d} "
              f"meanDice={summary['meanDice']:.3f} meanIoU={summary['meanIoU']:.3f} "
              f"(Dice@0.5={summary['Dice@0.5']:.3f})")

    pd.DataFrame(summaries).to_csv(out / f"{args.model}_summary.csv", index=False)
    pd.concat(per_image).to_csv(out / f"{args.model}_per_image.csv", index=False)


if __name__ == "__main__":
    main()

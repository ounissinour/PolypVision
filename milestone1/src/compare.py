"""Compare reproduced mean Dice / mean IoU with the paper tables.

Paper values (meanDic, meanIoU):
  PraNet paper (arXiv 2006.11392), Tables 1 and 2
  Polyp-PVT paper (arXiv 2108.06932), Tables 1-3; it re-reports PraNet and gives
  0.712 on CVC-ColonDB where the PraNet paper says 0.709 (both kept below).

Usage: python src/compare.py --results results/
"""

import argparse
import os

import pandas as pd

PAPER = {
    # (model, dataset): (meanDice, meanIoU, source)
    ("PraNet", "Kvasir"):               (0.898, 0.840, "PraNet paper Tab. 1"),
    ("PraNet", "CVC-ClinicDB"):         (0.899, 0.849, "PraNet paper Tab. 1"),
    ("PraNet", "CVC-ColonDB"):          (0.709, 0.640, "PraNet paper Tab. 2 (Polyp-PVT paper: 0.712)"),
    ("PraNet", "ETIS-LaribPolypDB"):    (0.628, 0.567, "PraNet paper Tab. 2"),
    ("PraNet", "CVC-300"):              (0.871, 0.797, "PraNet paper Tab. 2"),
    ("Polyp-PVT", "Kvasir"):            (0.917, 0.864, "Polyp-PVT paper Tab. 1"),
    ("Polyp-PVT", "CVC-ClinicDB"):      (0.937, 0.889, "Polyp-PVT paper Tab. 1"),
    ("Polyp-PVT", "CVC-ColonDB"):       (0.808, 0.727, "Polyp-PVT paper Tab. 2 / README"),
    ("Polyp-PVT", "ETIS-LaribPolypDB"): (0.787, 0.706, "Polyp-PVT paper Tab. 2"),
    ("Polyp-PVT", "CVC-300"):           (0.900, 0.833, "Polyp-PVT paper Tab. 3"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    ours = pd.concat(
        pd.read_csv(os.path.join(args.results, f"{m}_summary.csv"))
        for m in ["PraNet", "Polyp-PVT"]
        if os.path.exists(os.path.join(args.results, f"{m}_summary.csv"))
    )
    paper = pd.DataFrame(
        [(m, d, dice, iou, src) for (m, d), (dice, iou, src) in PAPER.items()],
        columns=["model", "dataset", "paper_meanDice", "paper_meanIoU", "source"],
    )
    t = ours.merge(paper, on=["model", "dataset"], how="left")
    t["dDice"] = t["meanDice"] - t["paper_meanDice"]
    t["dIoU"] = t["meanIoU"] - t["paper_meanIoU"]
    cols = ["model", "dataset", "n", "meanDice", "paper_meanDice", "dDice",
            "meanIoU", "paper_meanIoU", "dIoU", "Dice@0.5", "source"]
    t = t[cols].round(3)
    t.to_csv(os.path.join(args.results, "comparison.csv"), index=False)
    with open(os.path.join(args.results, "comparison.md"), "w") as f:
        f.write(t.to_markdown(index=False))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()

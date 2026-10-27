"""Put our mean Dice / mean IoU next to the values printed in the two papers.

Paper values come from the PraNet paper (arXiv 2006.11392, Tables 1-2) and the Polyp-PVT
paper (arXiv 2108.06932, Tables 1-3). The Polyp-PVT paper re-reports PraNet on CVC-ColonDB
as 0.712 where the PraNet paper says 0.709; we keep the PraNet paper's value and note the other.

Usage: python src/compare.py --results results
Reads <results>/<model>_summary.csv, writes <results>/comparison.csv and comparison.md.
"""

import argparse
from pathlib import Path

import pandas as pd

MODELS = ["PraNet", "Polyp-PVT"]

# (model, dataset): (meanDice, meanIoU, where it comes from)
PAPER = {
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

COLUMNS = ["model", "dataset", "n", "meanDice", "paper_meanDice", "dDice",
           "meanIoU", "paper_meanIoU", "dIoU", "Dice@0.5", "source"]


def compare(results_dir):
    """Return the comparison table for every <model>_summary.csv found in results_dir."""
    summaries = [results_dir / f"{m}_summary.csv" for m in MODELS]
    ours = pd.concat(pd.read_csv(p) for p in summaries if p.exists())
    paper = pd.DataFrame(
        [(model, dataset, dice, iou, source) for (model, dataset), (dice, iou, source) in PAPER.items()],
        columns=["model", "dataset", "paper_meanDice", "paper_meanIoU", "source"],
    )
    table = ours.merge(paper, on=["model", "dataset"], how="left")
    table["dDice"] = table["meanDice"] - table["paper_meanDice"]
    table["dIoU"] = table["meanIoU"] - table["paper_meanIoU"]
    return table[COLUMNS].round(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results)
    table = compare(results_dir)
    table.to_csv(results_dir / "comparison.csv", index=False)
    (results_dir / "comparison.md").write_text(table.to_markdown(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

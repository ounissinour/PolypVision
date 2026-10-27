"""Checks for src/evaluate.py and src/compare.py that need no data download.

Run from the milestone1 folder:  python -m pytest tests
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

M1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M1 / "src"))

from compare import compare  # noqa: E402
from evaluate import evaluate_dataset  # noqa: E402


def write_mask(path, mask):
    """Save a boolean array as a 0/255 PNG, like the test scripts and the datasets do."""
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


@pytest.fixture
def folders(tmp_path):
    gt_dir, pred_dir = tmp_path / "masks", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    return gt_dir, pred_dir


def square(size=32, top=8, left=8, h=16, w=16):
    mask = np.zeros((size, size), bool)
    mask[top:top + h, left:left + w] = True
    return mask


def test_perfect_prediction(folders):
    gt_dir, pred_dir = folders
    write_mask(gt_dir / "a.png", square())
    write_mask(pred_dir / "a.png", square())
    summary, per_image = evaluate_dataset(pred_dir, gt_dir)

    assert summary["n"] == 1
    assert summary["Dice@0.5"] == 1.0 and summary["IoU@0.5"] == 1.0
    # mean over 256 thresholds: the lowest threshold keeps every pixel, so it is just under 1
    assert 0.99 < summary["meanDice"] < 1.0
    assert list(per_image["image"]) == ["a.png"]


def test_prediction_twice_as_large_as_the_polyp(folders):
    gt_dir, pred_dir = folders
    write_mask(gt_dir / "a.png", square(w=16))
    write_mask(pred_dir / "a.png", square(left=0, w=32))  # covers the polyp plus the same area again
    summary, _ = evaluate_dataset(pred_dir, gt_dir)

    assert summary["Dice@0.5"] == pytest.approx(2 / 3)   # 2*16*16 / (16*16 + 16*32)
    assert summary["IoU@0.5"] == pytest.approx(1 / 2)    # 16*16 / (16*32)
    assert summary["meanDice"] == pytest.approx(2 / 3, abs=0.01)


def test_mean_over_images(folders):
    gt_dir, pred_dir = folders
    for name in ["a", "b"]:
        write_mask(gt_dir / f"{name}.png", square())
    write_mask(pred_dir / "a.png", square())             # perfect
    write_mask(pred_dir / "b.png", square(left=0, w=32))  # Dice 2/3
    summary, per_image = evaluate_dataset(pred_dir, gt_dir)

    assert summary["n"] == 2
    assert summary["Dice@0.5"] == pytest.approx((1 + 2 / 3) / 2)
    assert per_image["dice_0.5"].tolist() == pytest.approx([1.0, 2 / 3])


def test_missing_prediction_is_an_error(folders):
    gt_dir, pred_dir = folders
    write_mask(gt_dir / "a.png", square())
    with pytest.raises(FileNotFoundError):
        evaluate_dataset(pred_dir, gt_dir)


def test_compare_reproduces_committed_table(tmp_path):
    """compare() on the committed summaries must give the committed comparison.csv."""
    for model in ["PraNet", "Polyp-PVT"]:
        (tmp_path / f"{model}_summary.csv").write_bytes((M1 / "results" / f"{model}_summary.csv").read_bytes())
    table = compare(tmp_path)
    expected = pd.read_csv(M1 / "results" / "comparison.csv")
    pd.testing.assert_frame_equal(table.reset_index(drop=True), expected)


def test_paper_values_are_reproduced():
    """Every reproduced mean Dice / IoU is within 0.005 of the paper, as the README claims."""
    table = pd.read_csv(M1 / "results" / "comparison.csv")
    assert len(table) == 10
    assert (table["dDice"].abs() <= 0.005).all(), table[["model", "dataset", "dDice"]]
    assert (table["dIoU"].abs() <= 0.005).all(), table[["model", "dataset", "dIoU"]]
    colondb = table[(table.model == "Polyp-PVT") & (table.dataset == "CVC-ColonDB")]
    assert colondb["meanDice"].item() == 0.808

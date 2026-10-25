"""Generate notebooks/M1_Reproduce.ipynb (keeps the notebook diff-able as Python)."""

import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})


md("""
# Milestone 1 – Reproduce PraNet and Polyp-PVT

Runs both official test scripts on the five test sets and computes mean Dice / mean IoU
with `pysodmetrics` (same definition as the papers' MATLAB toolbox).
Runtime: GPU (T4 is enough). About 15 minutes end to end, most of it downloads.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
""")

md("Settings. Set `USE_DRIVE = True` to cache downloads and save results in your Google Drive.")
code("""
REPO_URL = "https://github.com/vaatus/polyp-m1-reproduce"
PRANET_COMMIT = "95f12bd4e32ec954b5069654abae7de9473fd1f5"
PVT_COMMIT = "cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4"
USE_DRIVE = True

import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/polyp-m1"
else:
    WORK = "/content/work"
# data and prediction maps stay on local disk: unzipping or writing thousands of small
# files on the Drive mount is slow and can stop half-way
DATA, MAPS = "/content/data", "/content/maps"
os.makedirs(WORK, exist_ok=True)
os.environ.update(WORK=WORK, DATA=DATA, MAPS=MAPS)
print(WORK)
""")

code("""
!pip install -q gdown pysodmetrics timm tabulate
""")

md("Clone this repository and the two official repositories at fixed commits.")
code("""
%cd /content
!rm -rf m1 PraNet Polyp-PVT
!git clone -q $REPO_URL m1
!git clone -q https://github.com/DengPingFan/PraNet && git -C PraNet checkout -q $PRANET_COMMIT
!git clone -q https://github.com/DengPingFan/Polyp-PVT && git -C Polyp-PVT checkout -q $PVT_COMMIT
!git -C PraNet log -1 --format='PraNet    %h %cd' && git -C Polyp-PVT log -1 --format='Polyp-PVT %h %cd'
""")

md("""
Download data and trained models (Google Drive links from the two READMEs).
Expected test-set sizes: CVC-300 60, CVC-ClinicDB 62, Kvasir 100, CVC-ColonDB 380, ETIS 196.
""")
code("""
!bash /content/m1/scripts/download.sh $WORK $DATA
""")

md("""
Compatibility patches. PraNet's `MyTest.py` uses `scipy.misc.imsave`, which no longer exists;
it is replaced by `cv2.imwrite(..., res*255)` exactly as Polyp-PVT's `Test.py` does.
See `scripts/patch_repos.py` for the full list; none of them change the predictions.
""")
code("""
!python /content/m1/scripts/patch_repos.py /content/PraNet /content/Polyp-PVT
!cd /content/PraNet && git diff --stat && cd /content/Polyp-PVT && git diff --stat
""")

md("Put data and weights where each test script expects them.")
code("""
%%bash
set -e
mkdir -p /content/PraNet/data /content/PraNet/snapshots/PraNet_Res2Net
ln -sfn $DATA/TestDataset /content/PraNet/data/TestDataset
ln -sf  $WORK/weights/PraNet-19.pth /content/PraNet/snapshots/PraNet_Res2Net/PraNet-19.pth

mkdir -p /content/Polyp-PVT/dataset /content/Polyp-PVT/model_pth /content/Polyp-PVT/pretrained_pth
ln -sfn $DATA/TestDataset /content/Polyp-PVT/dataset/TestDataset
ln -sf  $WORK/weights/PolypPVT.pth  /content/Polyp-PVT/model_pth/PolypPVT.pth
ln -sf  $WORK/weights/pvt_v2_b2.pth /content/Polyp-PVT/pretrained_pth/pvt_v2_b2.pth
ls -l /content/PraNet/snapshots/PraNet_Res2Net /content/Polyp-PVT/model_pth /content/Polyp-PVT/pretrained_pth
""")

md("Run the official test scripts (default test size 352).")
code("""
!cd /content/PraNet && python -W ignore MyTest.py
!cd /content/Polyp-PVT && python -W ignore Test.py
""")

code("""
%%bash
mkdir -p $MAPS
rm -rf $MAPS/PraNet $MAPS/Polyp-PVT
cp -r /content/PraNet/results/PraNet $MAPS/PraNet
cp -r /content/Polyp-PVT/result_map/PolypPVT $MAPS/Polyp-PVT
for m in PraNet Polyp-PVT; do
  for s in CVC-300 CVC-ClinicDB Kvasir CVC-ColonDB ETIS-LaribPolypDB; do
    printf '%-10s %-18s %4s maps\\n' $m $s $(ls $MAPS/$m/$s | wc -l)
  done
done
""")

md("""
Evaluate with pysodmetrics. `meanDice`/`meanIoU` are the paper metrics
(mean over 256 thresholds); `Dice@0.5` is a single 0.5 threshold, shown only for reference.
""")
code("""
%cd /content/m1
!mkdir -p results
!python src/evaluate.py --model PraNet    --pred_root $MAPS/PraNet    --gt_root $DATA/TestDataset --out results
!python src/evaluate.py --model Polyp-PVT --pred_root $MAPS/Polyp-PVT --gt_root $DATA/TestDataset --out results
""")

md("""
Evaluator check: score the authors' own published prediction maps (README links) with the same code.
This should give the paper numbers exactly; `results/authors_maps_check/` holds the expected output.
""")
code("""
%%bash
cd $WORK/downloads
[ -f pranet_maps.zip ] || gdown -q 1CJ6CTUdenumgiKXieuKXFohefRJwyFPY -O pranet_maps.zip
[ -f pvt_maps.zip ]    || gdown -q 1L0pFFmd9fbqnJnBwrM9cEV5nlRS32mbQ -O pvt_maps.zip
rm -rf /content/authors_maps && mkdir -p /content/authors_maps
unzip -q -o pranet_maps.zip -d /content/authors_maps && unzip -q -o pvt_maps.zip -d /content/authors_maps
cd /content/m1 && mkdir -p results/authors_maps_check
python src/evaluate.py --model PraNet    --pred_root /content/authors_maps/PraNet --gt_root $DATA/TestDataset --out results/authors_maps_check
python src/evaluate.py --model Polyp-PVT --pred_root /content/authors_maps/result_map/PolypPVT --gt_root $DATA/TestDataset --out results/authors_maps_check
""")

md("Compare our maps with the paper tables.")
code("""
!python src/compare.py --results results
""")

code("""
import pandas as pd, matplotlib.pyplot as plt, numpy as np
t = pd.read_csv("results/comparison.csv")
order = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, metric in zip(axes, ["Dice", "IoU"]):
    x = np.arange(len(order))
    for k, (model, c) in enumerate([("PraNet", "tab:blue"), ("Polyp-PVT", "tab:orange")]):
        d = t[t.model == model].set_index("dataset").loc[order]
        off = (k - 0.5) * 0.4
        ax.bar(x + off, d[f"mean{metric}"], 0.38, color=c, label=f"{model} (ours)")
        ax.scatter(x + off, d[f"paper_mean{metric}"], color="k", marker="_", s=300, zorder=3,
                   label="paper" if k == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15)
    ax.set_ylim(0.5, 1.0); ax.set_title(f"mean {metric}"); ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("figures/reproduced_vs_paper.png", dpi=200); plt.show()
""")

md("Qualitative check: the images where the two models disagree most on each dataset.")
code("""
import cv2
pi = pd.read_csv("results/PraNet_per_image.csv").merge(
    pd.read_csv("results/Polyp-PVT_per_image.csv"), on=["dataset", "image"], suffixes=("_pranet", "_pvt"))
pi["gap"] = pi["dice_mean_thr_pvt"] - pi["dice_mean_thr_pranet"]
rows = pi.loc[pi.groupby("dataset")["gap"].apply(lambda s: s.abs().idxmax())]
D = f"{DATA}/TestDataset"
fig, ax = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)))
for r, (_, row) in enumerate(rows.iterrows()):
    stem = os.path.splitext(row.image)[0]
    img_path = next(p for p in os.listdir(f"{D}/{row.dataset}/images") if os.path.splitext(p)[0] == stem)
    gt_path = next(p for p in os.listdir(f"{D}/{row.dataset}/masks") if os.path.splitext(p)[0] == stem)
    ims = [cv2.imread(f"{D}/{row.dataset}/images/{img_path}")[:, :, ::-1],
           cv2.imread(f"{D}/{row.dataset}/masks/{gt_path}", 0),
           cv2.imread(f"{MAPS}/PraNet/{row.dataset}/{row.image}", 0),
           cv2.imread(f"{MAPS}/Polyp-PVT/{row.dataset}/{row.image}", 0)]
    titles = [f"{row.dataset}/{row.image}", "GT",
              f"PraNet Dice {row.dice_mean_thr_pranet:.2f}", f"Polyp-PVT Dice {row.dice_mean_thr_pvt:.2f}"]
    for c in range(4):
        ax[r, c].imshow(ims[c], cmap=None if c == 0 else "gray"); ax[r, c].set_title(titles[c], fontsize=9); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig("figures/largest_disagreements.png", dpi=150); plt.show()
""")

md("Save results, figures and our prediction maps to `WORK` (Drive) so they survive the session.")
code("""
!mkdir -p $WORK/m1_out && cp -r results figures $MAPS $WORK/m1_out/ && ls $WORK/m1_out $WORK/m1_out/results
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).resolve().parent.parent / "notebooks" / "M1_Reproduce.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}")

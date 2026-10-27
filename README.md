# PolypVision

Automatic polyp segmentation in colonoscopy images: a comparison of
[PraNet](https://github.com/DengPingFan/PraNet) (CNN, MICCAI 2020) and
[Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) (transformer, 2023).

BME · MSc Deep Learning (VITMMA19) · Fall 2026

## Team

**Team name:** PolypVision

| Member | Neptun code |
|---|---|
| Nour Ounissi | RSPN69 |
| Taha Mersni | K1TO74 |
| Baha Amar | ZXQA2V |

Who does what is in [`docs/Team_Tasks.md`](docs/Team_Tasks.md).

## Project description

Polyp segmentation predicts a pixel-level binary mask of the polyp in a colonoscopy frame. We compare
two representative models from the same research line, PraNet (convolutional, reverse attention) and
Polyp-PVT (Pyramid Vision Transformer backbone). Both are trained on Kvasir-SEG + CVC-ClinicDB and
tested on five public sets (Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, CVC-300) with mean
Dice and mean IoU, so they can be compared fairly.

Research question: how do a CNN and a transformer compare in segmentation accuracy, generalisation to
data from other sources, robustness with few labels, and sensitivity to the evaluation protocol?

| Milestone | What | Status | Where |
|---|---|---|---|
| **1 · Reproduce** | Run both official test scripts on the five test sets with the authors' trained weights, score with pysodmetrics, compare with the paper tables; explore the data and prepare train / validation / test loaders | **done** | [`milestone1/`](milestone1/) |
| 2 · Improve | Retrain Polyp-PVT with a proper train/validation/test protocol; losses, augmentations, image sizes | planned | `notebooks/03_PolypPVT_Retraining.ipynb` |
| 3 · Contribute | Other-hospital data, 10/25/50 % labels, failure analysis, evaluation-choice audit | planned | `notebooks/04`–`06` |

Full project outline: [`docs/Project_Documentation.md`](docs/Project_Documentation.md).

## Files in the repository

```
README.md                          this file
requirements.txt                   packages to install on top of Colab's defaults
docs/
  Project_Documentation.md         project idea, research question, approach
  Team_Tasks.md                    members and task split
data/README.md                     dataset notes and the folder layout the models expect (data itself is not committed)
milestone1/
  M1_Reproduce.ipynb               THE notebook of Milestone 1, self-contained (see "How to run")
  prepare_data.py                  Python script: data exploration, train / validation / test split, PyTorch datasets and loaders
  README.md                        Milestone 1 write-up: method, results, PraNet/Kvasir investigation, data preparation
  requirements.txt                 what the notebook pip-installs (gdown, pysodmetrics, timm, tabulate)
  results/PraNet_summary.csv       mean Dice / IoU per test set, our maps     (same for Polyp-PVT_summary.csv)
  results/PraNet_per_image.csv     per-image scores                            (same for Polyp-PVT_per_image.csv)
  results/comparison.csv|md        ours vs paper, side by side
  results/authors_maps_check/      evaluator check: the authors' published maps scored with our code
  results/splits.csv               every image with its split (train / val / test), source, size and polyp size
  figures/reproduced_vs_paper.png  bar chart, ours vs paper
  figures/largest_disagreements.png images where the two models disagree most
  figures/data_samples.png         sample images with mask outlines from every split and source
  figures/data_stats.png           polyp size distribution and images per split
notebooks/03..06_*.ipynb           empty notebooks for Milestones 2 and 3 (title only)
src/metrics.py                     shared pysodmetrics wrapper for Milestones 2 and 3
src/evaluation|preprocessing|training/   empty, for Milestones 2 and 3
results/, figures/                 outputs of Milestones 2 and 3 (empty for now)
```

## Data source and how it is downloaded

All data and weights come from the Google Drive links published in the official READMEs of
[PraNet](https://github.com/DengPingFan/PraNet) and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT).
The download cell of the Milestone 1 notebook fetches them with `gdown`; nothing is committed to
this repository.

| What | Content | From |
|---|---|---|
| `TestDataset.zip` | the five test sets: Kvasir 100, CVC-ClinicDB 62, CVC-ColonDB 380, ETIS-LaribPolypDB 196, CVC-300 60 images + masks | PraNet README |
| `TrainDataset.zip` | Kvasir-SEG 900 + CVC-ClinicDB 550 training images + masks (not needed for Milestone 1) | PraNet README |
| `PraNet-19.pth` | PraNet trained weights | PraNet README |
| `PolypPVT.pth` | Polyp-PVT trained weights | Polyp-PVT README §4.7 |
| `pvt_v2_b2.pth` | PVTv2-B2 ImageNet backbone (Polyp-PVT's constructor loads it even at test time) | Polyp-PVT README §4.3 |
| authors' prediction maps | used only to check our evaluator | both READMEs |

The cell caches the zips and weights in a work folder (your Google Drive when `USE_DRIVE = True`),
unzips to local disk, removes `__MACOSX` folders, and fails loudly if any test set does not have the
expected number of images and masks. One copy of `TestDataset` is shared by both models, so they are
scored against identical masks.

Original dataset pages: [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/),
[CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) (research and education use only).
Pretrained weights are the authors' own, unchanged.

**Preparation for training.** `milestone1/prepare_data.py` holds out 10 % of `TrainDataset` as a
validation set (Kvasir at random, CVC-ClinicDB in blocks of consecutive video frames), keeps the five
test sets as the papers use them, and builds PyTorch loaders that return 3×352×352 normalised image
tensors as inputs and 1×352×352 binary masks as outputs (train 1285, val 165, test 798 images). Details
and data figures in [`milestone1/README.md`](milestone1/README.md).

## How to run

The solution is one Jupyter notebook, **`milestone1/M1_Reproduce.ipynb`**, plus the Python script
**`milestone1/prepare_data.py`** that the notebook fetches for its last section (so the notebook needs
nothing else from this repository).

1. Open it in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ounissinour/PolypVision/blob/main/milestone1/M1_Reproduce.ipynb)
2. Runtime → Change runtime type → GPU (a T4 is enough).
3. Runtime → Run all. The second cell asks to mount Google Drive (set `USE_DRIVE = False` to skip).

About 15 minutes end to end, most of it downloads. The notebook clones the two official repositories
at fixed commits, downloads data and weights, patches the test scripts, runs `MyTest.py` (PraNet) and
`Test.py` (Polyp-PVT), evaluates the prediction maps, scores the authors' published maps as a check,
prints the comparison with the paper tables, draws the figures, and finally explores the data and
builds the train / validation / test loaders. Results and figures are written to `/content/m1` and
copied to the work folder.

To run locally instead, install `requirements.txt` plus PyTorch and OpenCV and run the cells in order
(every step is plain Python or bash). A full CPU run on a Mac takes about 30 minutes, mostly
Polyp-PVT inference.

## Milestone 1 results

Mean Dice / mean IoU on the five test sets, authors' trained models, Colab T4. Paper value in
brackets; full write-up, per-image scores and figures in [`milestone1/README.md`](milestone1/README.md).

| Dataset (n) | PraNet | Polyp-PVT |
|---|---|---|
| Kvasir (100) | 0.894 (0.898) / 0.836 (0.840) | 0.917 (0.917) / 0.864 (0.864) |
| CVC-ClinicDB (62) | 0.899 (0.899) / 0.849 (0.849) | 0.937 (0.937) / 0.889 (0.889) |
| CVC-ColonDB (380) | 0.712 (0.709) / 0.640 (0.640) | **0.808** (0.808) / 0.727 (0.727) |
| ETIS-LaribPolypDB (196) | 0.628 (0.628) / 0.567 (0.567) | 0.787 (0.787) / 0.706 (0.706) |
| CVC-300 (60) | 0.871 (0.871) / 0.797 (0.797) | 0.900 (0.900) / 0.833 (0.833) |

Polyp-PVT matches the paper on every set, including the 0.808 target on CVC-ColonDB. PraNet matches
on four; on Kvasir we get 0.894 instead of 0.898 with the released checkpoint (investigated in the
Milestone 1 README). Scoring the authors' own published prediction maps with our evaluator gives all
ten paper values, so the metric is the papers' metric: mean Dice over 256 thresholds, not Dice at a
single 0.5 threshold, which is 0.003–0.009 higher on every set. An independent CPU re-run gives the
same ten numbers.

## References

- Fan et al., *PraNet: Parallel Reverse Attention Network for Polyp Segmentation*, MICCAI 2020 — [arXiv:2006.11392](https://arxiv.org/abs/2006.11392)
- Dong et al., *Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers*, 2021/2023 — [arXiv:2108.06932](https://arxiv.org/abs/2108.06932)
- Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset* — [arXiv:1911.07069](https://arxiv.org/abs/1911.07069) · [dataset](https://datasets.simula.no/kvasir-seg/)
- CVC-ClinicDB — [polyp.grand-challenge.org](https://polyp.grand-challenge.org/CVCClinicDB/)
- Evaluation audit (2026) — [arXiv:2607.08203](https://arxiv.org/abs/2607.08203)
- Optional: [MedSAM](https://github.com/bowang-lab/MedSAM)

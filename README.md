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
| **2 · Improve** | Retrain Polyp-PVT choosing the model on a validation split (not on the test sets); compare loss functions, augmentations and image sizes; optional MedSAM with box prompts | **code ready, runs on Colab pending** | [`milestone2/`](milestone2/) |
| 3 · Contribute | Other-hospital data, 10/25/50 % labels, failure analysis, evaluation-choice audit | planned | `milestone3/` (to be created) |

Full project outline: [`docs/Project_Documentation.md`](docs/Project_Documentation.md).

## Files in the repository

```
README.md                          this file
requirements.txt                   packages to install on top of Colab's defaults (for a local run)
docs/
  Project_Documentation.md         project idea, research question, approach
  Team_Tasks.md                    members and task split
milestone1/
  M1_Reproduce.ipynb               THE notebook of Milestone 1, self-contained (see "How to run")
  prepare_data.py                  data exploration, train / validation / test split, PyTorch datasets and loaders
  README.md                        Milestone 1 write-up: method, results, PraNet/Kvasir investigation, data preparation
  results/PraNet_summary.csv       mean Dice / IoU per test set, our maps     (same for Polyp-PVT_summary.csv)
  results/PraNet_per_image.csv     per-image scores                            (same for Polyp-PVT_per_image.csv)
  results/comparison.csv|md        ours vs paper, side by side
  results/authors_maps_check/      evaluator check: the authors' published maps scored with our code
  results/splits.csv               every image with its split (train / val / test), source, size and polyp size
  figures/                         ours vs paper, largest disagreements, data samples, data statistics
milestone2/
  M2_Improve.ipynb                 THE notebook of Milestone 2: retraining experiments + optional MedSAM
  polyp_m2.py                      training / evaluation code used by the notebook (losses, augmentations, train loop)
  README.md                        Milestone 2 write-up: protocol, experiments, findings, how to run
  results/, figures/               outputs of the Colab runs (added after the runs)
```

Data, weights, checkpoints and prediction maps are never committed (see `.gitignore`); the notebooks
download them and keep them on Google Drive.

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

Each milestone is one Colab notebook that fetches what it needs from this repository by itself.

**Milestone 1:** **`milestone1/M1_Reproduce.ipynb`**, plus the Python script
**`milestone1/prepare_data.py`** that the notebook fetches for its last section.

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

**Milestone 2:** **`milestone2/M2_Improve.ipynb`**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ounissinour/PolypVision/blob/main/milestone2/M2_Improve.ipynb),
same three steps (GPU runtime, Run all). It trains Polyp-PVT several times, so it needs several Colab
sessions: every run is saved to Drive after each epoch and resumes after a disconnect, and finished runs
are skipped, so just *Run all* again. `RUN_ONLY` splits the experiments between team members. Details in
[`milestone2/README.md`](milestone2/README.md).

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

## Milestone 2 – Improve

The official `Train.py` evaluates the five **test** sets after every epoch and keeps the epoch with the
best test Dice, so the released Polyp-PVT model was chosen on the test data. Milestone 2 retrains
Polyp-PVT with the Milestone 1 validation split (1285 train / 165 validation images): the epoch is
chosen on validation and the test sets are scored once. On top of the official recipe we compare,
one factor at a time:

| Factor | Options |
|---|---|
| Loss | official structure loss (as the code computes it), structure loss as the paper describes it, BCE + Dice, Dice |
| Augmentation | none (official default), flips, flips + rotation (official option), strong (+ zoom crop, colour, blur) |
| Image size | 256, 352 (official), 448 |

The best option per factor (on validation) is then trained for 100 epochs next to the official recipe,
and compared with the authors' model and PraNet. Optional: MedSAM with box prompts (ground-truth boxes,
perturbed boxes, and boxes from Polyp-PVT's prediction).

While building it we found that the official "weighted BCE" is a plain BCE: `structure_loss` passes
`reduce='none'`, an old boolean argument, so PyTorch averages the BCE to one number before the edge
weighting is applied. The notebook demonstrates this and tests the loss as the paper describes it.

The code was tested on CPU: with the authors' weights, `polyp_m2.evaluate` reproduces the
Milestone 1 numbers on all 798 test images (per-image difference < 1e-5). **The training runs need a Colab GPU and have not been
run yet**; results and figures will be added to `milestone2/results` and `milestone2/figures`.

## References

- Fan et al., *PraNet: Parallel Reverse Attention Network for Polyp Segmentation*, MICCAI 2020 — [arXiv:2006.11392](https://arxiv.org/abs/2006.11392)
- Dong et al., *Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers*, 2021/2023 — [arXiv:2108.06932](https://arxiv.org/abs/2108.06932)
- Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset* — [arXiv:1911.07069](https://arxiv.org/abs/1911.07069) · [dataset](https://datasets.simula.no/kvasir-seg/)
- CVC-ClinicDB — [polyp.grand-challenge.org](https://polyp.grand-challenge.org/CVCClinicDB/)
- Evaluation audit (2026) — [arXiv:2607.08203](https://arxiv.org/abs/2607.08203)
- Optional: [MedSAM](https://github.com/bowang-lab/MedSAM)

# Milestone 1 – Reproduce PraNet and Polyp-PVT

Reproduces the published mean Dice and mean IoU of
[PraNet](https://github.com/DengPingFan/PraNet) (CNN, MICCAI 2020) and
[Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) (transformer, 2023) on the five standard
test sets, using the authors' trained models and the same evaluation definition as their papers.

## Run it

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ounissinour/PolypVision/blob/main/milestone1/M1_Reproduce.ipynb)

`M1_Reproduce.ipynb` is self-contained: open it in Colab with a GPU runtime and run all cells.
It does not need anything else from this repository. It

1. clones both official repositories at fixed commits,
2. downloads the test data and the trained weights with `gdown` (Google Drive links from the two READMEs),
3. patches the official test scripts for a current software stack (`scipy.misc.imsave` → `cv2.imwrite`, …),
4. runs `MyTest.py` (PraNet) and `Test.py` (Polyp-PVT) to write the prediction maps,
5. computes mean Dice / mean IoU with pysodmetrics, on our maps and, as a check, on the authors' published maps,
6. compares them with the paper tables and draws the figures,
7. explores the data and prepares it for training with `prepare_data.py` (fetched from this repository):
   train / validation / test split and PyTorch loaders.

About 15 minutes on a T4, most of it downloads. The outputs of our run are committed here:
`results/` (`*_summary.csv`, `*_per_image.csv`, `comparison.csv|md`, `authors_maps_check/`, `splits.csv`)
and `figures/`.

`prepare_data.py` is also a stand-alone script. After the download step of the notebook (or any folder
with the same `TrainDataset/` and `TestDataset/` layout):

```
python prepare_data.py --data /content/data --out /content/m1
```

## Targets

meanDic / meanIoU from the papers (the notebook's comparison cell holds the sources):

| Dataset (n) | PraNet | Polyp-PVT |
|---|---|---|
| Kvasir (100) | 0.898 / 0.840 | 0.917 / 0.864 |
| CVC-ClinicDB (62) | 0.899 / 0.849 | 0.937 / 0.889 |
| CVC-ColonDB (380) | 0.709 / 0.640 ¹ | **0.808** / 0.727 |
| ETIS-LaribPolypDB (196) | 0.628 / 0.567 | 0.787 / 0.706 |
| CVC-300 (60) | 0.871 / 0.797 | 0.900 / 0.833 |

¹ The Polyp-PVT paper re-reports PraNet on ColonDB as 0.712.

## Reproduced results

Colab, Tesla T4, torch 2.11, official test scripts at test size 352. Mean Dice / mean IoU, paper value
in brackets (full table: `results/comparison.md`):

| Dataset (n) | PraNet | Polyp-PVT |
|---|---|---|
| Kvasir (100) | **0.894** (0.898) / **0.836** (0.840) | 0.917 (0.917) / 0.864 (0.864) |
| CVC-ClinicDB (62) | 0.899 (0.899) / 0.849 (0.849) | 0.937 (0.937) / 0.889 (0.889) |
| CVC-ColonDB (380) | 0.712 (0.709 ¹) / 0.640 (0.640) | **0.808** (0.808) / 0.727 (0.727) |
| ETIS-LaribPolypDB (196) | 0.628 (0.628) / 0.567 (0.567) | 0.787 (0.787) / 0.706 (0.706) |
| CVC-300 (60) | 0.871 (0.871) / 0.797 (0.797) | 0.900 (0.900) / 0.833 (0.833) |

![reproduced vs paper](figures/reproduced_vs_paper.png)

Polyp-PVT reproduces exactly on all five sets, including the 0.808 target on CVC-ColonDB. Polyp-PVT
beats PraNet on every set, most on the unseen ones: +0.096 Dice on CVC-ColonDB and +0.159 on ETIS.

A second, independent run of the whole pipeline on CPU (macOS, torch 2.14) gives the same ten
numbers to three decimals; per-image Dice differs from the T4 run by at most 1.3e-4.

PraNet reproduces exactly on four sets. On Kvasir we get 0.894 instead of 0.898:

- Our maps and the authors' published PraNet maps score the same on CVC-300, ClinicDB, ColonDB and
  ETIS image by image (per-image Dice difference < 0.0001).
- On Kvasir, 92 of 100 images agree; 8 differ by more than 0.01 Dice, two of them by 0.13 and 0.16
  (`cju2mh8t6p07008350e01tx2a`, `cju1amqw6p8pw0993d9gc5crl`).
- The test images and masks are pixel-identical to the original Kvasir-SEG release, re-running PraNet
  on CPU gives the same 0.894 as the T4, and feeding the original JPEGs instead of the PNGs also gives
  0.894. The differing images have no common size.

So the gap is not from the data, the metric or our hardware. Our best explanation is that the authors'
published Kvasir maps were produced with a slightly different checkpoint or software setup than the
released `PraNet-19.pth`; we could not confirm this.

## Evaluator check

Scoring the authors' published prediction maps (PraNet README §3.3, Polyp-PVT README §4.8) with the
notebook's evaluation cell reproduces all ten paper values to three decimals
(`results/authors_maps_check/`). The one difference is PraNet on CVC-ColonDB: 0.712, which is
what the Polyp-PVT paper reports; the PraNet paper says 0.709. So any gap between our maps and the
papers comes from inference, not from the metric.

**Single-threshold Dice.** `Dice@0.5` (one 0.5 threshold) is 0.003 to 0.009 higher than the papers'
mean Dice on every set and model. A hand-written 0.5-threshold metric therefore looks like a slight
improvement over the paper when it is not.

## Data exploration and preparation for training

The official code trains on `TrainDataset` (Kvasir-SEG 900 + CVC-ClinicDB 550 images) and has no
validation set. `prepare_data.py` builds one for Milestone 2 and leaves the five test sets as the
papers use them (`results/splits.csv`, one row per image with its split, source and paths):

| Split | Kvasir | CVC-ClinicDB | CVC-ColonDB | ETIS | CVC-300 | Total |
|---|---|---|---|---|---|---|
| train | 810 | 475 | | | | 1285 |
| val | 90 | 75 | | | | 165 |
| test | 100 | 62 | 380 | 196 | 60 | 798 |

10 % of `TrainDataset` is held out. Kvasir images are independent and are sampled at random. CVC-ClinicDB
frames come from videos, and consecutive frame numbers belong to the same video, so the hold-out is made
of whole blocks of 25 consecutive frames: a validation frame is never the neighbour of a training frame
(apart from block boundaries), which keeps the validation score honest.

What the data looks like (`figures/data_samples.png`, `figures/data_stats.png`): Kvasir images come in
hundreds of sizes (most around 620×530), CVC-ClinicDB is 384×288, CVC-ColonDB and CVC-300 574×500,
ETIS 1225×966. Masks are binary 0/255. Polyps are small: the median polyp covers 11 % of a Kvasir image,
7 % of a CVC-ClinicDB frame, 4 % on ColonDB and 2 % on ETIS, which has the smallest polyps and is the
hardest test set for both models.

![data samples](figures/data_samples.png)

**Final output: training, validation and test inputs and outputs.** `PolypDataset` returns, per image,
an input tensor 3×352×352 (resized, ImageNet-normalised, as both papers do) and an output tensor
1×352×352 with values 0/1. `make_loaders` gives the three loaders; the training one shuffles and adds
random flips. Batch of 16: inputs `(16, 3, 352, 352)`, outputs `(16, 1, 352, 352)`.

## Notes on method

**What "mean Dice" means.** The MATLAB toolbox both papers use (`PraNet/eval/main.m`) binarises each
min-max normalised map at 256 thresholds, computes Dice per image and threshold, averages over images,
then over thresholds. The notebook gets the same number from pysodmetrics (`FmeasureV2` with
`DICEHandler`/`IOUHandler`, dynamic mode) and checks the metric on two synthetic masks before using
it. Dice after a single 0.5 threshold is a different, usually higher number; we report it as
`Dice@0.5` only for reference. This matters for the 2026 audit question in Milestone 3.

**Patches** (one notebook cell), all compatibility only:
- PraNet `MyTest.py`: `scipy.misc.imsave` (removed from SciPy) → `cv2.imwrite(path, res*255)`,
  as Polyp-PVT's `Test.py` already does; `F.upsample` → `F.interpolate`; weights loaded with `map_location`.
- PraNet `lib/PraNet_Res2Net.py`: `pretrained=False` for the Res2Net backbone, whose ImageNet weights are
  loaded from the authors' local path; `PraNet-19.pth` contains the backbone weights anyway.
- Polyp-PVT `Test.py`: drop the unused `scipy.misc` import, `F.upsample` → `F.interpolate`.
  Its constructor still loads `pretrained_pth/pvt_v2_b2.pth`, so the backbone is downloaded too.

**Data.** One copy of `TestDataset` (from the PraNet README link) is shared by both models, so they are
scored against identical masks. CVC-ClinicDB is used only with the authors' fixed split; if it is ever
re-split, split by video sequence, not by image.

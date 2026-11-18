# Milestone 2 – Improve: retrain Polyp-PVT with a proper validation split

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ounissinour/PolypVision/blob/main/milestone2/M2_Improve.ipynb)

> **Status:** code done and tested on CPU (see *Checks*); the GPU training runs on Colab are still to
> do. The tables below fill in from `results/` once the notebook has run.

## Why retrain

The official Polyp-PVT `Train.py` has no validation set. After every epoch it scores the five **test**
sets and saves the epoch with the best test Dice as `PolypPVT.pth`. The released model, and the 0.808
we reproduced in Milestone 1, are therefore chosen on the test data. That number is optimistic: the
test sets were used to make a decision.

Here the epoch is chosen on the **validation split from Milestone 1** (`prepare_data.make_split`, seed
0: 1285 train / 165 validation images; Kvasir at random, CVC-ClinicDB in blocks of 25 consecutive
video frames). The five test sets (798 images) are scored **once**, with the chosen checkpoint.

## What the notebook does

| Section | Content |
|---|---|
| 1–2 | Official Polyp-PVT code (same pinned commit as M1), `polyp_m2.py`, data and weights (re-uses the M1 Drive cache) |
| 3 | The split, checked to be identical to `milestone1/results/splits.csv` |
| 4 | Checks: our inference + metric reproduce the paper numbers with the authors' model; the official loss bug; augmentation preview |
| 5 | 9 training runs, one factor at a time, 30 epochs each |
| 6 | Ablation table, training curves, *how optimistic is choosing the epoch on the test set?* |
| 7 | Final model: best option per factor (on validation), 100 epochs, next to the official recipe for 100 epochs |
| 8 | Optional: MedSAM with bounding-box prompts |

### Training setup (baseline = the official recipe)

PolypPVT model class unchanged, PVTv2-B2 ImageNet backbone, AdamW lr 1e-4 (constant, as the official
schedule never decays within 100 epochs), weight decay 1e-4, batch 16, gradient clamp 0.5, multi-scale
training (every batch at 0.75×, 1× and 1.25× the size), loss on both outputs P1 and P2, test map = P1 + P2.
Differences from `Train.py`: validation-based checkpoint choice, mixed precision (fp16) to fit Colab
time limits, resumable runs, a fixed seed.

### Experiments

| Factor | Runs (baseline in bold) |
|---|---|
| Loss | **`structure_official`**, `structure_weighted`, `bce_dice`, `dice` |
| Augmentation | **`none`**, `flip`, `flip_rotate`, `strong` |
| Image size (train = test) | `256`, **`352`**, `448` |

- `structure_official`: what the official code computes (plain BCE + edge-weighted IoU, see below).
- `structure_weighted`: the loss as the PraNet / Polyp-PVT papers describe it (edge-weighted BCE + edge-weighted IoU).
- `bce_dice`, `dice`: common segmentation losses (soft Dice per image).
- `flip_rotate` is the official `--augmentation True` (flips + rotation up to ±90°); `strong` adds a
  70–100 % zoom-in crop, brightness / contrast / saturation / hue jitter and Gaussian blur.

### The official loss is not what the paper says

`structure_loss` in `Train.py` (copied from PraNet) calls
`F.binary_cross_entropy_with_logits(pred, mask, reduce='none')`. `reduce` is a deprecated *boolean*
argument; the string `'none'` is truthy, so PyTorch returns the **mean** BCE over all pixels, one
number. `(weit * wbce).sum() / weit.sum()` then gives the same number back. So the edge-weighted BCE of
the paper is in practice an unweighted BCE; only the IoU term is edge-weighted. The notebook shows it
(output shape `()`, equal to the plain mean BCE), and `loss_structure_weighted` tests the intended loss.

## Results

*To be filled from `results/ablation.md`, `results/protocol_gap.csv`, `results/final_comparison.md`
and `results/medsam_comparison.md` after the Colab runs.*

| Run | val | Kvasir | ClinicDB | ColonDB | ETIS | CVC-300 | mean5 |
|---|---|---|---|---|---|---|---|
| baseline (official recipe, 30 ep) | | | | | | | |
| … | | | | | | | |

Reference (Milestone 1): authors' model, epoch chosen on test: Kvasir 0.917, ClinicDB 0.937, ColonDB
0.808, ETIS 0.787, CVC-300 0.900. PraNet: 0.894, 0.899, 0.712, 0.628, 0.871.

## Checks done

- **Inference + metric:** `polyp_m2.evaluate` (prediction exactly as `Test.py`, Dice / IoU exactly as
  M1) with the authors' `PolypPVT.pth` on the 798 real test images gives Kvasir 0.917, ClinicDB 0.937,
  ColonDB 0.808, ETIS 0.787, CVC-300 0.900; per-image Dice differs from Milestone 1 by at most 1e-5
  (CPU run; the notebook repeats this check on Colab and stops if it fails).
- **Module:** `polyp_m2.py` was run on CPU on a small synthetic dataset with the real PolypPVT
  architecture: all losses and augmentations, training with multi-scale + validation, interruption
  after epoch 1 and resume, skipping a finished run, test scoring + saved maps, `collect`, and the
  MedSAM functions (random weights). All Python cells of the notebook parse; a full end-to-end
  execution of the notebook itself was not completed locally, so the first Colab run is its real test.

## How to run

1. Open `M2_Improve.ipynb` in Colab, GPU runtime (T4), Run all. It mounts Google Drive.
2. Runs are saved to `MyDrive/polyp-m2/runs/<name>/` after every epoch. After a disconnect, Run all
   again: it resumes the unfinished run and skips finished ones.
3. To share the work, set `RUN_ONLY` to a subset (e.g. one factor per team member), then copy the run
   folders into one `runs/` folder before section 6.
4. Copy `MyDrive/polyp-m2/m2_out/results` and `figures` into this folder and fill the tables above.

`polyp_m2.py` contains everything the notebook trains and evaluates with (datasets and augmentations,
losses, `Config`, `train`, `evaluate`, `collect`, MedSAM helpers).

## Limitations

- One seed per configuration: differences under ~0.005 Dice are not meaningful. The final choice
  ignores gains below `MIN_GAIN = 0.005` on validation for this reason.
- One factor at a time: interactions (e.g. strong augmentation helping only with longer training) are
  not tested; the ablations are 30 epochs, the final runs 100.
- The validation split comes from the same two datasets as the training data, so it measures
  in-distribution performance; ColonDB, ETIS and CVC-300 remain the only out-of-distribution test.
- MedSAM with ground-truth boxes is semi-automatic (the box comes from the answer), so it is an upper
  bound for box prompting, not a fair competitor to Polyp-PVT.

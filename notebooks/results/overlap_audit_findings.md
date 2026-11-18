# Milestone 2 post-selection overlap audit

The audit was run only after the validation selection had been locked to `size_288` (epoch 100). It compared the five official test datasets with the official training-pool manifest; test images were not used for training, tuning, or checkpoint selection.

## Automated findings

- 49 test images were flagged as near-duplicate candidates by a 64-bit difference-hash search (nearest Hamming distance at most 3).
- No candidate had an exact SHA-256 match.
- The dataset and distance counts are in [`overlap_audit_counts.csv`](overlap_audit_counts.csv); all 49 nearest train-image pairs are in [`overlap_audit_candidates.csv`](overlap_audit_candidates.csv).

## Visual review

All seven side-by-side contact sheets embedded in the final Colab notebook were reviewed. Several CVC-ClinicDB test/train pairs show the same lesion and camera view with small frame-to-frame changes (examples include test/train stems `353/354`, `429/436`, `266/259`, and `481/480`). These are consistent with neighboring frames from the same video sequence. The available archive does not provide verified sequence IDs, so sequence membership cannot be established from filenames alone. Several cross-dataset hash candidates (including CVC-ColonDB versus ClinicDB/Kvasir) are visibly different images, confirming that a low dHash distance by itself is not proof of duplication.

## Evaluation status and implication

The final test evaluator did not produce metrics. An earlier call stopped at the overlap guard; the latest notebook attempt raised `NameError: json is not defined` before entering the evaluator. The locked validation checkpoint and validation results were not changed. Given the apparent ClinicDB sequence overlap, test scores should not be presented as an independent sequence-level benchmark until sequence membership is resolved.

Next, obtain a verified CVC-ClinicDB video-to-frame mapping and exclude any training frames from test sequences before retraining the controlled comparisons. If a reliable mapping is unavailable, use a documented training pool that does not include ClinicDB and rerun all comparisons consistently. Keep the existing five test datasets out of any new training and selection.

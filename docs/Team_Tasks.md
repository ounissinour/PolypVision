# PolypVision — Team Tasks

## Team Members

- Nour Ounissi
- Taha Mersni
- Baha Amar

---

## Nour Ounissi

### Main Responsibilities
- Project organization and repository management
- PraNet reproduction
- Evaluation pipeline
- Dice and IoU calculation
- Results organization
- Integration of the different experiments

### Tasks
1. Prepare and maintain the GitHub repository.
2. Set up the first Colab environment.
3. Reproduce PraNet using the official implementation and pretrained model.
4. Run inference on the selected test datasets.
5. Calculate Dice and IoU.
6. Compare reproduced results with the published results.
7. Organize final results and figures.
8. Help integrate the work of all group members.

---

## Taha Mersni

### Main Responsibilities
- Polyp-PVT reproduction
- Model and dataset setup
- Inference experiments
- Reproduction results

### Tasks
1. Study the official Polyp-PVT repository.
2. Set up Polyp-PVT in Google Colab.
3. Download and prepare the required pretrained model and datasets.
4. Run inference on the selected test datasets.
5. Generate prediction masks.
6. Calculate and record the evaluation results.
7. Compare reproduced results with the results reported in the paper.
8. Document any implementation or compatibility issues.

---

## Baha Amar

### Main Responsibilities
- Polyp-PVT retraining
- Training experiments
- Validation and checkpointing
- Training analysis

### Tasks
1. Prepare the Polyp-PVT training environment in Google Colab.
2. Prepare the training and validation datasets.
3. Implement a proper train/validation/test protocol.
4. Retrain Polyp-PVT.
5. Monitor training and validation losses.
6. Save the best checkpoint according to validation performance.
7. Experiment with relevant training configurations.
8. Record training results and generate training curves.

---

# Shared Tasks

All team members will collaborate on:

- Dataset preparation and verification
- Generalization experiments
- Limited-label experiments
- Failure analysis
- Result interpretation
- Figures and visual comparisons
- Final report
- Presentation
- GitHub organization

## Collaboration Rules

- Each member should document their experiments.
- Results should be saved in the appropriate `results/` folder.
- Figures should be saved in the appropriate `figures/` folder.
- Changes to shared code should be communicated to the group.
- The final results should be reviewed by all team members before inclusion in the report.4
## References

### Code

* [PraNet — Official Repository](https://github.com/DengPingFan/PraNet)
* [Polyp-PVT — Official Repository](https://github.com/DengPingFan/Polyp-PVT)
* [MedSAM — Official Repository](https://github.com/bowang-lab/MedSAM) *(optional)*

### Datasets

* [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
* [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) *(research and education use only)*

### Papers

* [PraNet](https://arxiv.org/abs/2006.11392)
* [Polyp-PVT](https://arxiv.org/abs/2108.06932)
* [Kvasir-SEG](https://arxiv.org/abs/1911.07069)
* [Evaluation Audit (2026)](https://arxiv.org/abs/2607.08203)

# Model Comparison

All numbers below are real, computed once on each model's own held-out
test split (never touched during training/tuning), via
`scripts/evaluate.py`. Raw artifacts: `runs/exp_a_baseline/test_metrics.json`,
`runs/exp_d_control/test_metrics.json`. 95% CIs are bootstrap percentile
intervals (1000 resamples).

## Experiment A — main task (COVID vs NORMAL, cross-source)

Baseline CNN (389,314 parameters), 30 epochs, never early-stopped (best
epoch 28, val_loss 0.0112). Test set: 280 images (249 NORMAL, 31 COVID).

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.9893 | 0.9750 – 1.0000 |
| Precision | 1.0000 | 1.0000 – 1.0000 |
| Sensitivity | 0.9032 | 0.7931 – 1.0000 |
| Specificity | 1.0000 | 1.0000 – 1.0000 |
| F1 | 0.9492 | 0.8846 – 1.0000 |
| ROC-AUC | 1.0000 | 1.0000 – 1.0000 |

Confusion matrix `[[TN=249, FP=0], [FN=3, TP=28]]`.

## Experiment D control — same-source task (NORMAL vs PNEUMONIA, Kermany only)

Identical architecture and training recipe. 23 epochs, early-stopped
(patience=5, best epoch 18, val_loss 0.1142). Test set: 926 images
(247 NORMAL, 679 PNEUMONIA).

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.9557 | 0.9417 – 0.9676 |
| Precision | 0.9775 | 0.9655 – 0.9876 |
| Sensitivity | 0.9617 | 0.9463 – 0.9752 |
| Specificity | 0.9393 | 0.9080 – 0.9663 |
| F1 | 0.9696 | 0.9596 – 0.9781 |
| ROC-AUC | 0.9920 | 0.9871 – 0.9956 |

Confusion matrix `[[TN=232, FP=15], [FN=26, TP=653]]`.

## The comparison, and why it matters more than either number alone

Experiment A's numbers look better — zero false positives, perfect
specificity, perfect ROC-AUC — but it should not be read as "the model is
better at detecting COVID than pneumonia." The two experiments differ in
exactly the variable this project set out to test: Experiment A's two
classes come from **different datasets** (Cohen vs Kermany); Experiment
D's two classes come from the **same** dataset, same hospital, same
imaging pipeline.

A same-architecture model making *zero* errors of one type across 249
same-direction test images (Experiment A's specificity) while a
comparable model on a same-source task makes a normal, mixed pattern of
15 false positives and 26 false negatives (Experiment D) is more
consistent with Experiment A exploiting a **dataset-source shortcut**
than with COVID vs. Normal being an intrinsically easier visual task than
Normal vs. Pneumonia. This is corroborated qualitatively by the Grad-CAM
analysis (`error_analysis.md`) — Experiment A's model repeatedly attends
to image borders and, in one case, an embedded text watermark, not lung
fields.

**This is not proof of causality.** Confounds remain: Experiment A's test
set is much smaller (31 vs. 679 positives, hence its wide sensitivity
CI), and COVID/pneumonia are not guaranteed to be equally hard tasks even
without any source confound. What this comparison shows is that
Experiment A's near-perfect numbers should not be taken at face value as
evidence of genuine diagnostic capability — which is the central,
intended finding of this project.

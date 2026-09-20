# Error Analysis & Grad-CAM Findings

## Method

For each model, predictions were collected on its real test split, then a
curated sample (correct-high-confidence, correct-low-confidence, false
negatives, false positives — as available per category) was selected via
`scripts/run_gradcam.py` and visualized with Grad-CAM
(`src/covidcxr/explain/gradcam.py`), hooked on the last conv block's
output. Overlays are saved under `reports/gradcam/{exp_a_baseline,
exp_d_control}/`. One overlay derived from a "No-Derivatives"-licensed
source image was generated during analysis and deleted before publishing
— it is not included here or in the repository (see `data/README.md` for
the licensing rule this enforces).

## Experiment A (COVID vs NORMAL) — qualitative findings

Visually inspecting the real overlays:

- **`correct_positive_high_conf_idx11...png`**: a hotspot sits directly on
  an **embedded text annotation** in the image corner (part of the
  original Cohen case-report scan), not on lung tissue.
- **`correct_positive_high_conf_idx16...png`**: attention *is* spread
  across the lung fields here — a more anatomically plausible case.
- **`false_negative_idx14...png`, `idx19...png`** (missed COVID cases):
  attention concentrated at the shoulders/collarbones and image corners,
  not centrally in the lungs — consistent with the model missing these
  cases in part because it wasn't looking at the lungs.
- **`correct_negative_high_conf_idx136...png`, `idx196...png`**: attention
  traces the rib margins and chest-wall periphery near the image edges,
  bilaterally symmetric, rather than the lung interior.

Pattern: multiple high-confidence correct predictions (and misses) are
driven substantially by image periphery / non-anatomical regions, not
lung parenchyma.

## Experiment D control (NORMAL vs PNEUMONIA) — comparison

- **`correct_positive_high_conf_idx230...png`** (pneumonia, correct):
  a large, central, lower/mid-lung attention region — anatomically where
  pneumonia consolidation commonly appears — alongside a recurring
  bottom-edge artifact strip present across several images in this set.
- **`correct_negative_high_conf_idx861...png`**: rib cage / upper-chest
  periphery, similar peripheral pull to Experiment A's NORMAL images
  (expected — same underlying Kermany image population).
- **`false_positive_idx19...png`**: the same recurring bottom artifact
  bar plus small round marker artifacts (plausibly ECG leads, common in
  pediatric films) draw attention alongside genuine rib-cage signal — a
  real, explainable error, not a border-only artifact.

## Live, out-of-distribution anecdote (informal, not part of the formal test set)

During manual use of the demo app, a chest X-ray sourced from outside
both training datasets (a genuinely external, real-world image, known by
the user to be normal) was classified as **COVID at 97.6% confidence**.
Its Grad-CAM overlay showed attention concentrated almost entirely on the
**far left and right vertical edges of the image** — essentially no
attention on the lung fields. This is consistent with (and a vivid,
independent illustration of) the same failure mode identified
quantitatively in Experiment D and qualitatively in the test-set Grad-CAM
review above: the model's border-padding behavior (from `ResizeAndPad`,
which letterboxes to a square) may itself be an exploitable, source-correlated
cue, since Cohen and Kermany images differ systematically in aspect
ratio and cropping. The image itself is not included in this repository
(external, rights/license unknown); this is a described observation, not
a redistributed asset.

## Conclusion

Two independent lines of evidence — the quantitative same-source vs.
cross-source comparison (`model_comparison.md`) and this qualitative
Grad-CAM review, corroborated by one live out-of-distribution failure —
point the same direction: Experiment A's near-perfect test metrics are
**more consistent with a dataset-source shortcut than with reliable
COVID detection**. This is the central, intended finding of this project
— not a limitation to be minimized, but the reason the project exists.

**Recommended follow-up** (not yet run): a controlled border/padding
sensitivity test — vary only the padding/crop of a fixed image and
observe whether the prediction changes — would isolate this mechanism
more directly than either experiment above.

# Findings

Notebooks are the primary artifact — this page is a map to them, not a substitute. Numbers below
are persisted in `results/*.json` and `results/*.csv`; figures in `figures/`.

## Validation strategy

330 samples come from 218 subjects (some sampled at multiple visits). Every split is grouped on
`subject_id` and stratified on label ([`splits.py`](src/biomedical_ml/splits.py)) — ignoring the
grouping inflates ROC-AUC by +0.002, small but real. More binding: at 22 control subjects (13:1
imbalance), a single 5-fold split gives fold-level ROC-AUC anywhere from 0.82 to 1.00 on an
identical model, so every headline number here is averaged over repeated grouped CV, never a
single split, and plain accuracy is never reported.

Details: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

## Does unsupervised structure track disease?

No. Raw-feature PCA does not separate SLE from controls (silhouette −0.01) even though a
supervised linear model reaches ROC-AUC ≈ 0.96 — the top-variance genes (IFI27, IFI44L, RSAD2,
IFIT1, OASL, OAS1 — SLE's interferon signature) vary strongly *within* the SLE group rather than
between SLE and controls, so they dominate variance without separating the classes.

Details: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) · `figures/pca_overview.png`

## Classical baselines

| Model | ROC-AUC | PR-AUC | Balanced accuracy |
|---|---|---|---|
| logreg_l2 (best) | 0.970 ± 0.061 | 0.997 ± 0.007 | 0.905 ± 0.050 |
| xgboost | 0.968 ± 0.062 | 0.996 ± 0.009 | 0.909 ± 0.094 |
| logreg_l1 | 0.958 ± 0.085 | 0.993 ± 0.014 | **0.944 ± 0.044** |
| logreg_elasticnet | 0.958 ± 0.084 | 0.994 ± 0.013 | 0.886 ± 0.054 |
| random_forest | 0.943 ± 0.087 | 0.991 ± 0.017 | 0.907 ± 0.082 |

All five are statistically indistinguishable. The best-to-worst gap is smaller than any model's
own fold-to-fold noise. Decision thresholds are tuned per-fold on training data only
(`evaluation._best_threshold`), since a blind 0.5 cutoff understates every model at this
imbalance.

SHAP's top genes for the best model share zero overlap with the top-variance genes above, and
aren't the interferon/HLA genes central to SLE biology, likely because 2000 collinear features
give an L2 model many equally-good ways to reach the same accuracy, not evidence of a novel axis.

Details: [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb)

## The autoencoder

Unsupervised MLP (2000 → 256 → 64 → **32** → 64 → 256 → 2000), trained on the same selected,
standardized features as the classical baselines, subject-grouped 80/20 split, early stopping
(val MSE 0.635, a 34% reduction over a trivial mean-prediction baseline).

Controlling for the fact that its input is already label-selected (unlike the raw
PCA above), three views separate feature-selection effects from compression effects:

| Representation | Silhouette |
|---|---|
| All 31,266 genes | −0.011 |
| 2000 selected genes, uncompressed | +0.196 |
| AE's 32-dim latent (compressed from above) | **+0.317** |

A linear probe on the 32-dim latent space matches the classical baselines, **ROC-AUC 0.961 ±
0.037**, the lowest variance of any model tested, using 1.6% of the features.

**Caveat:** this figure pools all 330 samples across CV folds, including some the AE's encoder
already saw (unlabeled) during its own training, an asymmetry the classical pipelines don't
share, since they refit feature selection per fold. Restricted to only the AE's held-out
subjects: ROC-AUC 0.936 (on just 4 control samples, too thin to be more authoritative than the
pooled figure, but a real, documented effect). 0.961 remains the headline number.

The genes behind the AE's separation,a coherent ribosomal/translation module (RPL18A, RPL11,
RPS27, EEF1B2...), overlap with neither the raw-variance genes nor the SHAP genes above. Caveat:
this kind of module often tracks immune cell-composition shifts rather than SLE-specific biology.

**Synthesis:** three methods, three non-overlapping gene sets, the same predictive ceiling. On
data this collinear, strong performance is compatible with genuine ambiguity about which genes
are responsible, a different representation can match performance through a completely
different route.

Details: [`notebooks/04_representation_comparison.ipynb`](notebooks/04_representation_comparison.ipynb) · `figures/latent_space_comparison.png`

## Design decisions

- **Autoencoder, not a classifier's hidden layer** — an MLP's hidden layer is optimized directly
  against the label, so asking "what did it discover" would be circular.
- **SHAP, not permutation importance** — signed, ranked attributions comparable against the
  latent space.
- **No feature selection outside the CV fold** — pre-filtering once, even on variance alone,
  leaks test-fold information.
- **No extra normalization** — the data ships already background-corrected and log2 transformed;
  re-normalizing would be cargo-culting.
- **CV evaluation generalized to any pipeline** — the latent-space probe reuses the exact same
  repeated grouped CV and threshold tuning as the classical baselines, not a parallel
  implementation.

A coverage audit (100% of `src/`, 129 tests) surfaced and fixed two real bugs (a SHAP fallback
path using a removed API, a batch-correlation function raising on single-sample groups), and a
reproducibility check in a clean environment caught a silent Jupyter kernel fallback, both
documented in commit history.

"""Computational helpers for the GSE138458 exploratory analysis.

Plotting and narrative live in the ``notebooks/`` walkthroughs; everything
here returns plain data so it can be tested and reused across notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from biomedical_ml.config import SEED
from biomedical_ml.preprocessing import BATCH_COLUMN, Dataset


def cohort_summary(dataset: Dataset) -> dict[str, object]:
    """Headline counts describing the cohort's sample/subject structure."""
    per_subject = dataset.groups.value_counts()
    controls = dataset.groups[dataset.y == 0]
    cases = dataset.groups[dataset.y == 1]

    return {
        "n_samples": int(dataset.X.shape[0]),
        "n_probes": int(dataset.X.shape[1]),
        "n_subjects": dataset.n_subjects,
        "n_sle_samples": int((dataset.y == 1).sum()),
        "n_control_samples": int((dataset.y == 0).sum()),
        "n_sle_subjects": int(cases.nunique()),
        "n_control_subjects": int(controls.nunique()),
        "samples_per_subject": {
            int(k): int(v) for k, v in per_subject.value_counts().sort_index().items()
        },
        "class_ratio": round(float((dataset.y == 1).sum() / max((dataset.y == 0).sum(), 1)), 2),
    }


def probe_variance(X: pd.DataFrame) -> pd.Series:
    """Per-probe variance across samples, sorted descending."""
    return X.var(axis=0).sort_values(ascending=False)


def pca_embedding(
    X: pd.DataFrame, n_components: int = 10, seed: int = SEED
) -> tuple[pd.DataFrame, np.ndarray]:
    """PCA on mean-centred expression.

    Probes are centred but deliberately *not* scaled to unit variance: on log2
    array data, scaling would inflate low-expression probes whose variance is
    mostly measurement noise.
    """
    n_components = min(n_components, *X.shape)
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(X.values - X.values.mean(axis=0, keepdims=True))

    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(scores, index=X.index, columns=columns), pca.explained_variance_ratio_


def batch_class_crosstab(metadata: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cross-tabulate BeadChip against class, to expose any batch/label confound."""
    labels = y.map({0: "control", 1: "sle"})
    return pd.crosstab(metadata[BATCH_COLUMN], labels)


def batch_confound_report(metadata: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Summarise how controls are distributed over BeadChips.

    If controls clustered on a few chips, any case/control signal would be
    partly a batch effect. This quantifies that risk.
    """
    crosstab = batch_class_crosstab(metadata, y)
    with_control = crosstab[crosstab.get("control", 0) > 0]

    return {
        "n_chips": int(crosstab.shape[0]),
        "n_chips_with_control": int(with_control.shape[0]),
        "max_controls_on_one_chip": int(crosstab.get("control", pd.Series(0)).max()),
        "chips_are_mixed": bool(
            (with_control.get("sle", pd.Series(dtype=int)) > 0).all() if len(with_control) else False
        ),
    }


def correlate_pcs_with_batch(
    scores: pd.DataFrame, metadata: pd.DataFrame, n_components: int = 5
) -> pd.DataFrame:
    """One-way ANOVA of each PC against BeadChip, as an eta-squared effect size.

    A PC that is strongly explained by chip is capturing batch rather than
    biology; this is the quantitative version of eyeballing a coloured scatter.
    """
    from scipy import stats

    rows = []
    chips = metadata[BATCH_COLUMN]
    for pc in scores.columns[:n_components]:
        groups = [g.to_numpy() for _, g in scores[pc].groupby(chips) if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_value = stats.f_oneway(*groups)

        values = scores[pc].to_numpy()
        grand_mean = values.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((values - grand_mean) ** 2).sum()

        rows.append(
            {
                "component": pc,
                "f_stat": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": float(ss_between / ss_total) if ss_total else np.nan,
            }
        )

    if not rows:
        # Every component was skipped (fewer than two multi-sample batches) --
        # pd.DataFrame([]).set_index("component") would KeyError since there
        # is no such column on an empty frame.
        return pd.DataFrame(columns=["f_stat", "p_value", "eta_squared"]).rename_axis("component")
    return pd.DataFrame(rows).set_index("component")


def correlate_features_with_target(
    X: pd.DataFrame,
    target: np.ndarray | pd.Series,
    annotation: pd.DataFrame | None = None,
    *,
    n: int = 20,
) -> pd.DataFrame:
    """Rank ``X``'s columns by absolute Pearson correlation with ``target``.

    A model-agnostic way to ask "which raw features track this direction" —
    used to connect a disease-associated autoencoder latent dimension back to
    genes (Day 4), since the encoder's non-linearities rule out reading gene
    importance directly off its weight matrices the way PCA loadings would.
    Works equally well for any other fixed numeric direction over the same
    samples ``X`` was measured on.
    """
    target = np.asarray(target, dtype=float)
    correlations = X.apply(lambda col: np.corrcoef(col.to_numpy(), target)[0, 1], axis=0)

    ranking = pd.DataFrame(
        {"probe_id": correlations.index, "correlation": correlations.to_numpy()}
    )
    ranking["abs_correlation"] = ranking["correlation"].abs()
    ranking = ranking.sort_values("abs_correlation", ascending=False)

    if annotation is not None:
        ranking["gene_symbol"] = ranking["probe_id"].map(annotation["gene_symbol"])

    return ranking.head(n).reset_index(drop=True)
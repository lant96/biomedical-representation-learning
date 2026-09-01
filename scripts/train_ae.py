"""Train the GSE138458 autoencoder and save its checkpoint and loss curve.

Re-run with different hyperparameters via CLI flags (which override
``configs/default.yaml`` rather than requiring the file to be edited, so runs
stay traceable — see the comment at the top of that file):

    python scripts/train_ae.py
    python scripts/train_ae.py --latent-dim 16
    python scripts/train_ae.py --max-epochs 500 --patience 30 --seed 7
    python scripts/train_ae.py --config configs/my_variant.yaml

Writes:
    results/ae_checkpoint.pt          model weights + feature space + split
    results/ae_training_summary.json  hyperparameters used, final losses
    figures/ae_loss_curve.png         train/val reconstruction loss per epoch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from biomedical_ml.ae_training import save_checkpoint, train_autoencoder
from biomedical_ml.config import (
    CONFIG_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
    load_config,
)
from biomedical_ml.preprocessing import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "default.yaml")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def plot_loss_curve(history: dict[str, list[float]], best_epoch: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train", color="#4c72b0")
    ax.plot(epochs, history["val_loss"], label="val", color="#c44e52")
    ax.axvline(best_epoch + 1, ls="--", lw=0.8, color="0.5", label=f"best epoch ({best_epoch + 1})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction loss (MSE)")
    ax.set_title("Autoencoder training: reconstruction loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    ensure_dirs()

    cfg = load_config(args.config)
    ae_cfg = cfg["autoencoder"]
    seed = args.seed if args.seed is not None else cfg["seed"]

    hyperparams = {
        "k": cfg["features"]["k"],
        "latent_dim": args.latent_dim if args.latent_dim is not None else ae_cfg["latent_dim"],
        "hidden_dims": tuple(ae_cfg["hidden_dims"]),
        "dropout": ae_cfg["dropout"],
        "weight_decay": ae_cfg["weight_decay"],
        "learning_rate": ae_cfg["learning_rate"],
        "batch_size": ae_cfg["batch_size"],
        "max_epochs": args.max_epochs if args.max_epochs is not None else ae_cfg["max_epochs"],
        "patience": args.patience if args.patience is not None else ae_cfg["patience"],
        "min_delta": ae_cfg["min_delta"],
        "n_splits": cfg["split"]["n_splits"],
        "seed": seed,
    }
    print("Training with:", json.dumps(hyperparams, default=list, indent=2))

    dataset = build_dataset(annotated_only=True)
    print(dataset.summary())

    result = train_autoencoder(dataset, **hyperparams)
    n_epochs_run = len(result.history["train_loss"])
    print(
        f"\nStopped after {n_epochs_run} epochs "
        f"(best epoch {result.best_epoch + 1}, val loss {result.best_val_loss:.4f})"
    )

    checkpoint_path = RESULTS_DIR / "ae_checkpoint.pt"
    save_checkpoint(result, dataset, checkpoint_path)
    print(f"wrote {checkpoint_path}")

    figure = plot_loss_curve(result.history, result.best_epoch)
    figure_path = FIGURES_DIR / "ae_loss_curve.png"
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {figure_path}")

    summary = {
        "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in hyperparams.items()},
        "n_epochs_run": n_epochs_run,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "final_train_loss": result.history["train_loss"][-1],
        "final_val_loss": result.history["val_loss"][-1],
        "n_train_samples": len(result.train_idx),
        "n_val_samples": len(result.val_idx),
        "n_train_subjects": len(set(dataset.groups.iloc[result.train_idx])),
        "n_val_subjects": len(set(dataset.groups.iloc[result.val_idx])),
    }
    summary_path = RESULTS_DIR / "ae_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

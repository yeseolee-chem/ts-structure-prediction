#!/usr/bin/env python
"""
plot_metrics.py — CSV → PNG plots for PyTorch Lightning CSVLogger output.

Usage
-----
    # Auto-find the most recent run under logs/
    python plot_metrics.py

    # Point at a specific CSV or run directory
    python plot_metrics.py --csv logs/<run_name>/<run_name>/version_0/metrics.csv
    python plot_metrics.py --run-dir logs/<run_name>

Outputs (PNG) are written next to the CSV under a `plots/` subfolder:
  - loss.png          : train / val loss curves
  - rmsd.png          : val RMSD mean & median (the structure-accuracy plot)
  - scaled_err.png    : val_ep_scaled_err (the EarlyStopping monitor)
  - lr.png            : learning-rate schedule (if logged)
  - summary.png       : 2x2 panel combining the above

Designed to run with only numpy + pandas + matplotlib (no wandb / tensorboard).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Non-interactive backend — works on headless SLURM nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------
def find_latest_metrics_csv(logs_root: Path) -> Optional[Path]:
    """Return the newest metrics.csv under logs_root, or None if none found."""
    candidates = list(logs_root.rglob("metrics.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_csv(args) -> Path:
    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        if not p.is_file():
            sys.exit(f"ERROR: CSV not found: {p}")
        return p
    if args.run_dir:
        root = Path(args.run_dir).expanduser().resolve()
        found = find_latest_metrics_csv(root)
        if not found:
            sys.exit(f"ERROR: no metrics.csv under {root}")
        return found
    # fallback: auto-find
    root = Path("logs").resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} does not exist. Run training first, or pass --csv.")
    found = find_latest_metrics_csv(root)
    if not found:
        sys.exit(f"ERROR: no metrics.csv under {root}")
    return found


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def _collapse_by_epoch(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Group rows that share the same `epoch` and average `col`.

    PL writes one row per logged step, so an epoch often spans several rows.
    Averaging within an epoch gives one point per epoch for clean plotting.
    """
    if "epoch" not in df.columns or col not in df.columns:
        return pd.DataFrame(columns=["epoch", col])
    sub = df[["epoch", col]].dropna()
    if sub.empty:
        return sub
    return sub.groupby("epoch", as_index=False)[col].mean()


def _plot_line(ax, df: pd.DataFrame, col: str, label: str, **kw) -> bool:
    data = _collapse_by_epoch(df, col)
    if data.empty:
        return False
    ax.plot(data["epoch"], data[col], label=label, **kw)
    return True


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------
def plot_loss(df: pd.DataFrame, out: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = False
    ok |= _plot_line(ax, df, "tr_loss", "train loss", color="tab:blue")
    ok |= _plot_line(ax, df, "val_ep_loss", "val loss",   color="tab:red")
    if not ok:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("Loss curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_rmsd(df: pd.DataFrame, out: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = False
    ok |= _plot_line(ax, df, "val_ep_rmsd_median", "val RMSD median (primary)",
                     color="tab:orange", linewidth=2)
    ok |= _plot_line(ax, df, "val_ep_rmsd_mean",   "val RMSD mean",
                     color="tab:green", linestyle="--")
    if not ok:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Structure accuracy — TS prediction vs. ground truth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_scaled_err(df: pd.DataFrame, out: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = False
    ok |= _plot_line(ax, df, "val_ep_scaled_err", "val scaled err (EarlyStop monitor)",
                     color="tab:purple", linewidth=2)
    ok |= _plot_line(ax, df, "tr_scaled_err",     "train scaled err",
                     color="tab:gray", linestyle=":")
    if not ok:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch")
    ax.set_ylabel("scaled error")
    ax.set_title("Scaled-error curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_lr(df: pd.DataFrame, out: Path) -> bool:
    # LR is step-logged; use step if available, else epoch.
    lr_cols = [c for c in df.columns if c.startswith("lr")]
    if not lr_cols:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_col = "step" if "step" in df.columns else "epoch"
    plotted = False
    for c in lr_cols:
        sub = df[[x_col, c]].dropna()
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub[c], label=c)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel(x_col)
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_summary(df: pd.DataFrame, out: Path) -> bool:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_loss, ax_rmsd), (ax_serr, ax_lr) = axes

    # loss
    ok_loss = False
    ok_loss |= _plot_line(ax_loss, df, "tr_loss", "train", color="tab:blue")
    ok_loss |= _plot_line(ax_loss, df, "val_ep_loss", "val", color="tab:red")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    if ok_loss:
        ax_loss.set_yscale("log"); ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # rmsd
    ok_rmsd = False
    ok_rmsd |= _plot_line(ax_rmsd, df, "val_ep_rmsd_median", "median (primary)",
                          color="tab:orange", linewidth=2)
    ok_rmsd |= _plot_line(ax_rmsd, df, "val_ep_rmsd_mean", "mean",
                          color="tab:green", linestyle="--")
    ax_rmsd.set_title("Structure accuracy (val RMSD, Å)")
    ax_rmsd.set_xlabel("epoch"); ax_rmsd.set_ylabel("RMSD (Å)")
    if ok_rmsd:
        ax_rmsd.legend()
    ax_rmsd.grid(True, alpha=0.3)

    # scaled err
    ok_serr = _plot_line(ax_serr, df, "val_ep_scaled_err",
                         "val scaled err", color="tab:purple", linewidth=2)
    ax_serr.set_title("EarlyStop monitor (val_ep_scaled_err)")
    ax_serr.set_xlabel("epoch"); ax_serr.set_ylabel("scaled err")
    if ok_serr:
        ax_serr.legend()
    ax_serr.grid(True, alpha=0.3)

    # lr
    lr_cols = [c for c in df.columns if c.startswith("lr")]
    ok_lr = False
    x_col = "step" if "step" in df.columns else "epoch"
    for c in lr_cols:
        sub = df[[x_col, c]].dropna()
        if sub.empty:
            continue
        ax_lr.plot(sub[x_col], sub[c], label=c)
        ok_lr = True
    ax_lr.set_title("LR schedule")
    ax_lr.set_xlabel(x_col); ax_lr.set_ylabel("lr")
    if ok_lr:
        ax_lr.legend()
    ax_lr.grid(True, alpha=0.3)

    fig.suptitle(f"Training summary — {out.parent.parent.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Best-epoch textual summary (handy when you can't open the PNGs remotely)
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    def _stat(col: str, how: str = "min"):
        data = _collapse_by_epoch(df, col)
        if data.empty:
            return None, None
        idx = data[col].idxmin() if how == "min" else data[col].idxmax()
        return float(data.loc[idx, col]), int(data.loc[idx, "epoch"])

    print("")
    print("====== Best-epoch summary ======")
    for col, label in [
        ("val_ep_rmsd_median", "val RMSD median (Å)"),
        ("val_ep_rmsd_mean",   "val RMSD mean (Å)"),
        ("val_ep_scaled_err",  "val scaled err"),
        ("val_ep_loss",        "val loss"),
    ]:
        val, epoch = _stat(col, "min")
        if val is None:
            print(f"  {label:30s} : (not logged)")
        else:
            print(f"  {label:30s} : {val:.5f}   @ epoch {epoch}")
    print("================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=None,
                        help="Explicit metrics.csv path.")
    parser.add_argument("--run-dir", default=None,
                        help="Search this dir (recursively) for metrics.csv.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write PNGs (default: <csv-dir>/plots/).")
    args = parser.parse_args(argv)

    csv_path = resolve_csv(args)
    print(f"CSV : {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("ERROR: metrics.csv is empty — no epochs completed yet.")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUT : {out_dir}")

    produced = []
    if plot_loss(df,        out_dir / "loss.png"):        produced.append("loss.png")
    if plot_rmsd(df,        out_dir / "rmsd.png"):        produced.append("rmsd.png")
    if plot_scaled_err(df,  out_dir / "scaled_err.png"):  produced.append("scaled_err.png")
    if plot_lr(df,          out_dir / "lr.png"):          produced.append("lr.png")
    if plot_summary(df,     out_dir / "summary.png"):     produced.append("summary.png")

    if not produced:
        print("WARNING: no plots produced — CSV has no recognised metric columns.")
        print(f"        columns available: {list(df.columns)}")
    else:
        print("Wrote  :")
        for name in produced:
            print(f"         {out_dir / name}")

    print_summary(df)


if __name__ == "__main__":
    main()

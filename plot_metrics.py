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
# Column we use to pick the single "best" epoch. EarlyStopping +
# ModelCheckpoint both monitor val_ep_scaled_err, so summarizing other
# metrics at THAT epoch describes the same checkpoint reviewers would
# evaluate. The previous behavior — picking idxmin per metric — could
# stitch together numbers from totally different epochs (e.g. RMSD median
# from epoch 47, RMSD mean from epoch 92), which made the reported summary
# correspond to no actual saved model.
MONITOR_COL = "val_ep_scaled_err"

_SUMMARY_COLS = [
    ("val_ep_rmsd_median", "val RMSD median (Å)"),
    ("val_ep_rmsd_mean",   "val RMSD mean (Å)"),
    ("val_ep_rmsd_std",    "val RMSD std (Å)"),
    ("val_ep_scaled_err",  "val scaled err"),
    ("val_ep_loss",        "val loss"),
]


def _best_epoch_rows(df: pd.DataFrame):
    """Report all summary metrics from the best-monitor epoch.

    Returns rows of (label, value, epoch). The chosen epoch is the one
    that minimizes ``MONITOR_COL`` — same metric EarlyStopping watches —
    so the reported numbers describe one consistent checkpoint state.
    """
    monitor_data = _collapse_by_epoch(df, MONITOR_COL)
    if monitor_data.empty:
        # Monitor column wasn't logged → degrade gracefully to per-metric
        # idxmin so the user still gets something. Mark the rows so the
        # caller can flag the inconsistency.
        rows = []
        for col, label in _SUMMARY_COLS:
            data = _collapse_by_epoch(df, col)
            if data.empty:
                rows.append((label, None, None))
            else:
                idx = data[col].idxmin()
                rows.append(
                    (label + " (per-metric idxmin — monitor missing)",
                     float(data.loc[idx, col]),
                     int(data.loc[idx, "epoch"]))
                )
        return rows

    best_idx = monitor_data[MONITOR_COL].idxmin()
    best_epoch = int(monitor_data.loc[best_idx, "epoch"])

    rows = []
    for col, label in _SUMMARY_COLS:
        data = _collapse_by_epoch(df, col)
        if data.empty:
            rows.append((label, None, None))
            continue
        # Look up this metric at the best-monitor epoch (not its own argmin).
        match = data[data["epoch"] == best_epoch]
        if match.empty:
            # Metric wasn't logged on that exact epoch (e.g. logged every 10
            # epochs); take the closest available epoch instead.
            closest_idx = (data["epoch"] - best_epoch).abs().idxmin()
            rows.append(
                (label,
                 float(data.loc[closest_idx, col]),
                 int(data.loc[closest_idx, "epoch"]))
            )
        else:
            rows.append(
                (label,
                 float(match.iloc[0][col]),
                 int(match.iloc[0]["epoch"]))
            )
    return rows


def _format_summary(rows, header: str) -> str:
    lines = [f"====== {header} ======"]
    lines.append(
        f"  (all metrics taken from the epoch that minimizes "
        f"{MONITOR_COL!r} — the monitor used by EarlyStopping/ModelCheckpoint)"
    )
    for label, val, epoch in rows:
        if val is None:
            lines.append(f"  {label:30s} : (not logged)")
        else:
            lines.append(f"  {label:30s} : {val:.5f}   @ epoch {epoch}")
    lines.append("=" * (len(header) + 14))
    return "\n".join(lines)


def print_summary(df: pd.DataFrame) -> None:
    rows = _best_epoch_rows(df)
    print("")
    print(_format_summary(rows, "Best-epoch summary"))


def write_summary_file(df: pd.DataFrame, out_path: Path, csv_path: Path) -> None:
    """Persist the best-epoch RMSD summary to a plain-text result file."""
    rows = _best_epoch_rows(df)
    run_label = csv_path.parent.parent.name  # logs/<run>/<run>/version_X/metrics.csv
    body = _format_summary(rows, f"RMSD summary — {run_label}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
        fh.write(f"\nSource CSV : {csv_path}\n")
    print(f"RESULT : {out_path}")


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

    # Persist the same RMSD summary to results/<run>_rmsd_summary.txt so there
    # is a proper "result file" on disk (paired with the PNGs under plots/).
    run_label = csv_path.parent.parent.name
    repo_root = Path(__file__).resolve().parent
    results_dir = repo_root / "results"
    summary_path = results_dir / f"{run_label}_rmsd_summary.txt"
    write_summary_file(df, summary_path, csv_path)


if __name__ == "__main__":
    main()

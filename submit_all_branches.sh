#!/bin/bash
# ===========================================================================
# submit_all_branches.sh — submit one SLURM training job per experimental branch
# ---------------------------------------------------------------------------
# Each experimental branch (cb-*, rp-*, reactot-halo8) implements a different
# idea. To run them in parallel without cross-branch source contamination
# (see reactot/trainer/train_rpsb_ts1x.py::_git_branch_tag), every job needs
# its own checked-out working directory — a `git worktree`.
#
# This wrapper:
#   1. For each branch, finds an existing worktree (anywhere) OR creates a
#      new one at "../ts-prediction-<B>".
#   2. Pulls the latest origin/<B> into that worktree.
#   3. Pre-creates logs/, checkpoint/, results/ in the worktree so that
#      `sbatch` can never fail to land in a half-set-up directory even if
#      the slurm script body never runs.
#   4. Submits run_${WRAPPER}_slurm.sh from the worktree with EXPERIMENT_ID=<B>.
#
# Why this exists:
#   The simpler one-liner
#     [ -d "$WT" ] || git worktree add "$WT" "$B"
#   fails with
#     fatal: '<B>' is already used by worktree at '<some-path>'
#   when <B> is already checked out somewhere else (most commonly the main
#   repo itself). In that situation `git worktree add` cannot proceed, the
#   target directory is never created, and the subsequent `cd "$WT"` aborts
#   the loop. We recover by querying `git worktree list --porcelain` for an
#   existing checkout and reusing its path.
#
# Usage:
#   bash submit_all_branches.sh                                 # mix, dl=1000
#   WRAPPER=halo8 DATA_LIMIT=300 bash submit_all_branches.sh    # halo8 wrapper
#   BRANCHES="cb-A cb-B" bash submit_all_branches.sh            # subset
#   DRY_RUN=1 bash submit_all_branches.sh                       # show plan only
# ===========================================================================
set -u

# ---- Configurable defaults ------------------------------------------------
BRANCHES=${BRANCHES:-"reactot-halo8 cb-A cb-AB cb-ABD cb-B cb-BC cb-BCD cb-C cb-CD cb-D rp-A rp-B rp-C rp-D rp-E rp-F"}
WRAPPER=${WRAPPER:-mix}                # halo8 | t1x | mix
DATA_LIMIT=${DATA_LIMIT:-1000}
DRY_RUN=${DRY_RUN:-0}

SLURM_SCRIPT="run_${WRAPPER}_slurm.sh"

# Resolve repo root from this script's location so the wrapper works no
# matter which directory the user invokes it from.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
PARENT_DIR=$(dirname "$REPO_ROOT")

echo "==========================================="
echo "submit_all_branches.sh"
echo "REPO_ROOT  : $REPO_ROOT"
echo "PARENT_DIR : $PARENT_DIR"
echo "WRAPPER    : $WRAPPER ($SLURM_SCRIPT)"
echo "DATA_LIMIT : $DATA_LIMIT"
echo "DRY_RUN    : $DRY_RUN"
echo "BRANCHES   : $BRANCHES"
echo "==========================================="

# Refresh remote refs once up-front; the per-branch `pull --ff-only` reads
# from this fetch.
if [ "$DRY_RUN" = "0" ]; then
    git -C "$REPO_ROOT" fetch --all --prune || echo "WARN: fetch failed; continuing"
fi

submit_count=0
fail_count=0
failed_branches=""

for B in $BRANCHES; do
    echo ""
    echo "=== branch: $B ==="

    # 1. Locate (or create) a worktree for this branch ----------------------
    EXISTING_WT=$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null \
        | awk -v b="refs/heads/$B" '
            /^worktree / { wt = substr($0, 10) }
            $0 == "branch " b { print wt; exit }
        ')

    if [ -n "$EXISTING_WT" ]; then
        WT="$EXISTING_WT"
        echo "Reusing existing worktree for $B at: $WT"
    else
        WT="$PARENT_DIR/ts-prediction-$B"
        if [ -d "$WT" ]; then
            echo "Path exists but is not a registered worktree: $WT"
            echo "  → trying 'git worktree repair' to re-register it"
            if ! git -C "$REPO_ROOT" worktree repair "$WT" 2>&1; then
                echo "ERROR: cannot register $WT as a worktree."
                echo "       Either remove it or rerun with BRANCHES excluding $B."
                fail_count=$((fail_count+1))
                failed_branches="$failed_branches $B"
                continue
            fi
        else
            echo "Creating worktree: $WT  →  $B"
            if [ "$DRY_RUN" = "1" ]; then
                echo "  (DRY_RUN: would run 'git worktree add $WT $B')"
            elif ! git -C "$REPO_ROOT" worktree add "$WT" "$B" 2>&1; then
                echo "ERROR: 'git worktree add $WT $B' failed; skipping"
                fail_count=$((fail_count+1))
                failed_branches="$failed_branches $B"
                continue
            fi
        fi
    fi

    # 2. Pull latest --------------------------------------------------------
    if [ "$DRY_RUN" = "0" ] && [ -d "$WT/.git" -o -f "$WT/.git" ]; then
        if ! git -C "$WT" pull --ff-only origin "$B" 2>&1; then
            echo "WARN: pull --ff-only failed for $B (diverged?). Continuing with current local state."
        fi
    fi

    # 3. Pre-create output directories -------------------------------------
    # The slurm wrappers also do this, but creating them here means the
    # checkpoint/log/result tree is in a known state before sbatch dispatch
    # — useful when sbatch itself rejects the job before the script body
    # ever runs.
    mkdir -p "$WT/logs" "$WT/checkpoint" "$WT/results"

    # 4. Verify the wrapper script exists in this branch -------------------
    if [ ! -f "$WT/$SLURM_SCRIPT" ]; then
        echo "ERROR: $SLURM_SCRIPT not found in $WT — skipping"
        fail_count=$((fail_count+1))
        failed_branches="$failed_branches $B"
        continue
    fi

    # 5. Submit -------------------------------------------------------------
    echo "Submitting: DATA_LIMIT=$DATA_LIMIT EXPERIMENT_ID=$B sbatch $SLURM_SCRIPT (cwd=$WT)"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  (DRY_RUN: not actually submitting)"
        submit_count=$((submit_count+1))
    elif ( cd "$WT" && DATA_LIMIT="$DATA_LIMIT" EXPERIMENT_ID="$B" sbatch "$SLURM_SCRIPT" ); then
        submit_count=$((submit_count+1))
    else
        echo "ERROR: sbatch failed for $B"
        fail_count=$((fail_count+1))
        failed_branches="$failed_branches $B"
    fi
done

echo ""
echo "==========================================="
echo "Summary: submitted=$submit_count  failed=$fail_count"
if [ -n "$failed_branches" ]; then
    echo "Failed branches:$failed_branches"
fi
echo "==========================================="

[ "$fail_count" -eq 0 ]

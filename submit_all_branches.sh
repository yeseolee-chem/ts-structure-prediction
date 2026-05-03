#!/bin/bash
# ===========================================================================
# submit_all_branches.sh — submit one SLURM training job per experimental branch
# ---------------------------------------------------------------------------
# Each experimental branch (cb-*, rp-*, reactot-halo8) implements a different
# idea. To run them in parallel without cross-branch source contamination
# (see reactot/trainer/train_rpsb_ts1x.py::_git_branch_tag), every job needs
# its own checked-out working directory — a `git worktree`.
#
# Default behavior (matches the documented manual submission flow):
#   1. For each branch in BRANCHES: locate an existing worktree (anywhere)
#      OR create a new one at "../ts-prediction-<B>"; pull origin/<B>;
#      pre-create logs/, checkpoint/, results/; submit run_${WRAPPER}_slurm.sh
#      with DATA_LIMIT=$DATA_LIMIT EXPERIMENT_ID=<B>.
#   2. Then, for each branch in EXTRA_T1X_BRANCHES (default
#      "reactot-halo8"), additionally submit run_t1x_slurm.sh — this is
#      the T1x-only baseline that pairs with the augmented mix dataset.
#
# Why this exists — the simpler inline loop that this replaces:
#     for B in $BRANCHES; do
#       WT="../ts-prediction-$B"
#       [ -d "$WT" ] || git worktree add "$WT" "$B"
#       ( cd "$WT" && sbatch run_mix_slurm.sh )
#     done
# fails for two distinct reasons that this wrapper handles:
#
#   - When <B> is already checked out somewhere (most commonly the main
#     repo itself, e.g. /gpfs/.../ts-structure-prediction holding cb-C),
#     `git worktree add` aborts with
#         fatal: '<B>' is already used by worktree at '<some-path>'
#     the target dir is never created, and the subsequent `cd` fails.
#     We recover by querying `git worktree list --porcelain` for the
#     existing checkout and reusing its path.
#
#   - SLURM's `#SBATCH --output=logs/<wrapper>_%j.out` directive is
#     resolved against cwd at submit time but writes the file at
#     job-start time. If logs/ doesn't exist when the job starts, the
#     stdout/stderr capture silently fails — and the `mkdir -p` inside
#     the slurm script body runs too late to fix it. We pre-create
#     logs/, checkpoint/, results/ in the worktree before sbatch.
#
# Usage:
#   bash submit_all_branches.sh                                 # mix (all) + t1x (halo8)
#   WRAPPER=t1x bash submit_all_branches.sh                     # t1x for the main pass
#   EXTRA_T1X_BRANCHES="" bash submit_all_branches.sh           # disable the extra pass
#   BRANCHES="cb-A cb-B" bash submit_all_branches.sh            # subset
#   DATA_LIMIT=300 bash submit_all_branches.sh                  # smaller subset
#   DRY_RUN=1 bash submit_all_branches.sh                       # show plan only
# ===========================================================================
set -u

# ---- Configurable defaults ------------------------------------------------
BRANCHES=${BRANCHES:-"reactot-halo8 cb-A cb-AB cb-ABD cb-B cb-BC cb-BCD cb-C cb-CD cb-D rp-A rp-B rp-C rp-D rp-E rp-F"}
WRAPPER=${WRAPPER:-mix}                # halo8 | t1x | mix
DATA_LIMIT=${DATA_LIMIT:-1000}
DRY_RUN=${DRY_RUN:-0}
# Branches that ALSO get a t1x submission after the main pass. Set to ""
# to disable. Default reactot-halo8 — see the dataset-vs-dataset comment
# in run_mix_slurm.sh / run_t1x_slurm.sh headers. Skipped automatically
# if WRAPPER=t1x already (the main pass covered it).
EXTRA_T1X_BRANCHES=${EXTRA_T1X_BRANCHES:-"reactot-halo8"}

# Resolve repo root from this script's location so the wrapper works no
# matter which directory the user invokes it from.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
PARENT_DIR=$(dirname "$REPO_ROOT")

echo "==========================================="
echo "submit_all_branches.sh"
echo "REPO_ROOT          : $REPO_ROOT"
echo "PARENT_DIR         : $PARENT_DIR"
echo "WRAPPER            : $WRAPPER (run_${WRAPPER}_slurm.sh)"
echo "DATA_LIMIT         : $DATA_LIMIT"
echo "DRY_RUN            : $DRY_RUN"
echo "BRANCHES           : $BRANCHES"
echo "EXTRA_T1X_BRANCHES : ${EXTRA_T1X_BRANCHES:-<none>}"
echo "==========================================="

# Refresh remote refs once up-front; the per-branch `pull --ff-only` reads
# from this fetch.
if [ "$DRY_RUN" = "0" ]; then
    git -C "$REPO_ROOT" fetch --all --prune || echo "WARN: fetch failed; continuing"
fi

submit_count=0
fail_count=0
failed_jobs=""

# ---- Locate or create worktree for branch ---------------------------------
# Prints worktree path on stdout. All diagnostics go to stderr so the
# caller can do `WT=$(ensure_worktree X)` cleanly. Returns 1 on failure.
ensure_worktree() {
    local B="$1"
    local existing wt

    existing=$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null \
        | awk -v b="refs/heads/$B" '
            /^worktree / { wt = substr($0, 10) }
            $0 == "branch " b { print wt; exit }
        ')

    if [ -n "$existing" ]; then
        echo "Reusing existing worktree for $B at: $existing" >&2
        echo "$existing"
        return 0
    fi

    wt="$PARENT_DIR/ts-prediction-$B"
    if [ -d "$wt" ]; then
        echo "Path exists but is not a registered worktree: $wt" >&2
        echo "  -> trying 'git worktree repair' to re-register it" >&2
        if ! git -C "$REPO_ROOT" worktree repair "$wt" >&2 2>&1; then
            echo "ERROR: cannot register $wt as a worktree." >&2
            echo "       Either remove it, or rerun with BRANCHES excluding $B." >&2
            return 1
        fi
    elif [ "$DRY_RUN" = "1" ]; then
        echo "(DRY_RUN: would run 'git worktree add $wt $B')" >&2
    elif ! git -C "$REPO_ROOT" worktree add "$wt" "$B" >&2 2>&1; then
        echo "ERROR: 'git worktree add $wt $B' failed; skipping" >&2
        return 1
    else
        echo "Created worktree: $wt -> $B" >&2
    fi

    echo "$wt"
    return 0
}

# ---- Submit one slurm wrapper for one branch ------------------------------
# Mutates submit_count / fail_count / failed_jobs.
submit_one() {
    local B="$1" SCRIPT="$2" LABEL="$3" WT
    echo ""
    echo "=== $LABEL: $B ($SCRIPT) ==="

    if ! WT=$(ensure_worktree "$B"); then
        fail_count=$((fail_count+1))
        failed_jobs="$failed_jobs $LABEL:$B"
        return 1
    fi

    # Pull latest (best-effort: divergence is a warning, not fatal).
    if [ "$DRY_RUN" = "0" ] && [ -e "$WT/.git" ]; then
        if ! git -C "$WT" pull --ff-only origin "$B" 2>&1; then
            echo "WARN: pull --ff-only failed for $B (diverged?). Continuing with current local state."
        fi
    fi

    # Pre-create output directories — see header for why this MUST happen
    # before sbatch. The slurm script body also creates them, but that's
    # too late: SLURM has already opened logs/<wrapper>_%j.out by then.
    mkdir -p "$WT/logs" "$WT/checkpoint" "$WT/results"

    if [ ! -f "$WT/$SCRIPT" ]; then
        echo "ERROR: $SCRIPT not found in $WT — skipping"
        fail_count=$((fail_count+1))
        failed_jobs="$failed_jobs $LABEL:$B"
        return 1
    fi

    echo "Submitting: DATA_LIMIT=$DATA_LIMIT EXPERIMENT_ID=$B sbatch $SCRIPT (cwd=$WT)"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  (DRY_RUN: not actually submitting)"
        submit_count=$((submit_count+1))
        return 0
    fi
    if ( cd "$WT" && DATA_LIMIT="$DATA_LIMIT" EXPERIMENT_ID="$B" sbatch "$SCRIPT" ); then
        submit_count=$((submit_count+1))
        return 0
    fi
    echo "ERROR: sbatch failed for $B ($SCRIPT)"
    fail_count=$((fail_count+1))
    failed_jobs="$failed_jobs $LABEL:$B"
    return 1
}

# ---- Main pass: WRAPPER for every branch in BRANCHES ----------------------
for B in $BRANCHES; do
    submit_one "$B" "run_${WRAPPER}_slurm.sh" "$WRAPPER"
done

# ---- Extra pass: t1x for paired-baseline branches -------------------------
if [ -n "$EXTRA_T1X_BRANCHES" ] && [ "$WRAPPER" != "t1x" ]; then
    for EB in $EXTRA_T1X_BRANCHES; do
        submit_one "$EB" "run_t1x_slurm.sh" "extra-t1x"
    done
fi

echo ""
echo "==========================================="
echo "Summary: submitted=$submit_count  failed=$fail_count"
if [ -n "$failed_jobs" ]; then
    echo "Failed jobs:$failed_jobs"
fi
echo "==========================================="

[ "$fail_count" -eq 0 ]

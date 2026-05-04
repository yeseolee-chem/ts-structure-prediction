#!/bin/bash
# ===========================================================================
# reset_and_resubmit.sh — wipe everything and resubmit the full Idea-1 matrix
# ---------------------------------------------------------------------------
# 1. scancel every running/queued job for ${USER:-${USERNAME:-unknown}} and wait for the queue to
#    drain (max 60 s).
# 2. Recursively delete logs/, checkpoint/, results/, wandb/ in EVERY
#    registered git worktree (and the main repo). Handles both real dirs
#    and the central-tree symlinks that submit_all_branches.sh sets up.
# 3. Recreate the central logs/checkpoint/results layout in the main repo.
# 4. Fetch + ff-only refresh on the main repo so the wrapper script picks
#    up the latest fix(slurm) commits before re-submission. Per-branch
#    pulls happen inside submit_all_branches.sh's loop.
# 5. Submit run_mix_slurm.sh for every branch in BRANCHES plus
#    run_t1x_slurm.sh for every branch in EXTRA_T1X_BRANCHES, all with
#    DATA_LIMIT=$DATA_LIMIT.
#
# Defaults give: reactot-halo8 + 9 cb-* (mix), plus a t1x submission for
# reactot-halo8 to provide the T1x-only baseline.
#
# Usage:
#   bash reset_and_resubmit.sh                  # interactive, asks for "yes"
#   YES=1 bash reset_and_resubmit.sh            # skip the confirmation prompt
#   DRY_RUN=1 bash reset_and_resubmit.sh        # print the plan, change nothing
#   DATA_LIMIT=1000 bash reset_and_resubmit.sh  # different subset size
#   BRANCHES="cb-B cb-BCD" bash reset_and_resubmit.sh  # subset
# ===========================================================================
set -u

BRANCHES=${BRANCHES:-"reactot-halo8 cb-A cb-AB cb-ABD cb-B cb-BC cb-BCD cb-C cb-CD cb-D"}
EXTRA_T1X_BRANCHES=${EXTRA_T1X_BRANCHES:-"reactot-halo8"}
DATA_LIMIT=${DATA_LIMIT:-500}
DRY_RUN=${DRY_RUN:-0}
YES=${YES:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
SUBMIT_SCRIPT="$REPO_ROOT/submit_all_branches.sh"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "ERROR: $SUBMIT_SCRIPT not found. The current branch must contain"
    echo "       submit_all_branches.sh (cb-AB does). 'cd' into the cb-AB"
    echo "       worktree before running this script."
    exit 1
fi

n_mix=$(echo "$BRANCHES" | wc -w)
n_t1x=$(echo "$EXTRA_T1X_BRANCHES" | wc -w)

echo "==========================================="
echo "reset_and_resubmit.sh"
echo "REPO_ROOT          : $REPO_ROOT"
echo "BRANCHES (mix)     : $BRANCHES"
echo "EXTRA_T1X_BRANCHES : $EXTRA_T1X_BRANCHES"
echo "DATA_LIMIT         : $DATA_LIMIT"
echo "DRY_RUN            : $DRY_RUN"
echo "Will submit        : ${n_mix} mix + ${n_t1x} t1x = $((n_mix + n_t1x)) jobs"
echo "==========================================="

if [ "$YES" != "1" ] && [ "$DRY_RUN" != "1" ]; then
    echo ""
    echo "This will:"
    echo "  - scancel every running/queued job for ${USER:-${USERNAME:-unknown}}"
    echo "  - DELETE logs/, checkpoint/, results/, wandb/ in every worktree"
    echo "  - resubmit ${n_mix} mix + ${n_t1x} t1x jobs (DATA_LIMIT=$DATA_LIMIT)"
    echo ""
    echo "Recovery of deleted output is impossible — make sure you have"
    echo "downloaded anything you want to keep."
    echo ""
    read -p "Type 'yes' to continue: " ans
    [ "$ans" = "yes" ] || { echo "Aborted."; exit 1; }
fi

# ---- 1. Cancel all jobs and wait for queue to drain ----------------------
echo ""
echo ">>> [1/5] scancel -u ${USER:-${USERNAME:-unknown}}"
if [ "$DRY_RUN" = "0" ]; then
    scancel -u "${USER:-${USERNAME:-unknown}}" 2>/dev/null || true
    for i in $(seq 1 60); do
        n=$(squeue -u "${USER:-${USERNAME:-unknown}}" -h 2>/dev/null | wc -l)
        [ "$n" -eq 0 ] && break
        echo "    waiting... $n job(s) still queued"
        sleep 1
    done
    n=$(squeue -u "${USER:-${USERNAME:-unknown}}" -h 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "WARN: $n job(s) still queued after 60s — continuing anyway"
    fi
fi

# ---- 2. Wipe outputs across every worktree -------------------------------
# rm -rf works for both symlinks (removes the link, leaves target alone)
# AND real dirs (recursive delete). We want the data gone wherever it
# lives, so iterate every worktree first, then the main repo, then
# recreate the central output dirs at the end.
echo ""
echo ">>> [2/5] Wiping logs/checkpoint/results/wandb across all worktrees"
WORKTREES=$(git -C "$REPO_ROOT" worktree list --porcelain \
            | awk '/^worktree / {print substr($0, 10)}')
deleted=0
for WT in $WORKTREES; do
    for D in logs checkpoint results wandb; do
        if [ -e "$WT/$D" ] || [ -L "$WT/$D" ]; then
            if [ "$DRY_RUN" = "0" ]; then
                rm -rf "$WT/$D"
            fi
            deleted=$((deleted + 1))
            echo "    rm -rf $WT/$D"
        fi
    done
done
echo "    ($deleted entries removed)"

# ---- 3. Recreate central output layout -----------------------------------
# submit_all_branches.sh's ensure_output_layout will recreate the per-
# worktree symlinks pointing here on the next submission.
echo ""
echo ">>> [3/5] Recreating central output dirs"
if [ "$DRY_RUN" = "0" ]; then
    mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoint" "$REPO_ROOT/results"
fi
echo "    mkdir -p $REPO_ROOT/{logs,checkpoint,results}"

# ---- 4. Refresh main repo (per-branch pull is done inside submit script) -
echo ""
echo ">>> [4/5] git fetch --all --prune"
if [ "$DRY_RUN" = "0" ]; then
    git -C "$REPO_ROOT" fetch --all --prune 2>&1 || echo "WARN: fetch failed; continuing"
fi

# ---- 5. Hand off to submit_all_branches.sh -------------------------------
echo ""
echo ">>> [5/5] submit_all_branches.sh"
cd "$REPO_ROOT"
BRANCHES="$BRANCHES" \
WRAPPER=mix \
DATA_LIMIT="$DATA_LIMIT" \
EXTRA_T1X_BRANCHES="$EXTRA_T1X_BRANCHES" \
DRY_RUN="$DRY_RUN" \
bash "$SUBMIT_SCRIPT"

echo ""
echo "==========================================="
echo "Done. Monitor with: squeue -u ${USER:-${USERNAME:-unknown}}"
echo "Tail output       : tail -f $REPO_ROOT/logs/mix_<jobid>.out"
echo "==========================================="

#!/bin/bash
# ===========================================================================
# cluster_setup.sh — one-shot "apply latest fix on cluster" runbook
# ---------------------------------------------------------------------------
# Run this on the UBAI cluster (gate1/gate2) AFTER ssh login. It encodes
# the entire "before sbatch" workflow so the user does not have to remember
# 10 separate `git pull`s and a preflight call.
#
# Typical session:
#
#   # On your laptop (Windows / Mac / Linux):
#   ssh -i ~/.ssh/yeseo1ee.pem yeseo1ee@172.16.xxx.xxx
#
#   # On the cluster (gate1):
#   cd ~/projects/ts-prediction-cb-AB   # cb-AB is the captain worktree where
#                                       # tools/ and submit_all_branches.sh live
#   bash tools/cluster_setup.sh         # pulls all branches + preflight
#   bash submit_all_branches.sh         # submit only if preflight passed
#
# What this script does:
#   1. `git fetch --all --prune` against the main repo to refresh remote refs.
#      The main repo's HEAD is left alone — every cb-* branch (including
#      cb-AB) lives in its own dedicated worktree at
#      $PARENT_DIR/ts-prediction-<branch>/.
#   2. For each cb-* / reactot-halo8 branch, ensures a dedicated worktree
#      lives at $PARENT_DIR/ts-prediction-<branch>/ and pulls origin/<branch>
#      into it.
#   3. Strips any __pycache__ that might shadow the freshly-pulled code (this
#      is also done inside run_*_slurm.sh at job start, but cleaning here
#      makes the preflight result honest).
#   4. Runs tools/preflight_check.sh and refuses to proceed if any branch
#      reports a contamination-class failure.
#
# Re-run this any time you (or another collaborator) push new commits to
# any of the 10 branches. Cheap (a few seconds when nothing has changed).
# ===========================================================================
set -u
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel) || {
    echo "ERROR: $SCRIPT_DIR is not in a git repo" >&2; exit 2
}
PARENT_DIR=$(dirname "$REPO_ROOT")
BRANCHES=${BRANCHES:-"reactot-halo8 cb-A cb-AB cb-ABD cb-B cb-BC cb-BCD cb-C cb-CD cb-D"}

echo "==========================================="
echo "cluster_setup.sh"
echo "REPO_ROOT  : $REPO_ROOT"
echo "PARENT_DIR : $PARENT_DIR"
echo "BRANCHES   : $BRANCHES"
echo "==========================================="

# ---- Step 1: refresh remote refs --------------------------------------------
echo ""
echo "[main repo] git fetch --all --prune..."
git -C "$REPO_ROOT" fetch --all --prune || echo "WARN: fetch failed; continuing with stale refs"

# ---- Step 2: per-branch worktree pulls --------------------------------------
echo ""
echo "[per-branch worktrees]"
fail_branches=""
for B in $BRANCHES; do
    WT="$PARENT_DIR/ts-prediction-$B"

    # Discover where git thinks $B lives.
    actual=$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null \
        | awk -v b="refs/heads/$B" '
            /^worktree / { wt = substr($0, 10) }
            $0 == "branch " b { print wt; exit }
        ')

    # If the branch is checked out elsewhere (notably the main repo, the
    # cb-C/612178 incident pattern), git worktree add will refuse. The
    # user must manually relocate it — print an actionable diagnostic.
    if [ -n "$actual" ] && [ "$actual" != "$WT" ]; then
        actual_real=$(cd "$actual" 2>/dev/null && pwd -P)
        repo_real=$(cd "$REPO_ROOT" 2>/dev/null && pwd -P)
        if [ "$actual_real" = "$repo_real" ]; then
            echo "  $B: ERROR — checked out IN main repo $REPO_ROOT" >&2
            echo "         (volatile HEAD; would contaminate parallel runs)" >&2
            echo "         Fix:  git -C $REPO_ROOT switch cb-AB" >&2
            echo "               git -C $REPO_ROOT worktree add $WT $B" >&2
            fail_branches="$fail_branches $B"
            continue
        else
            echo "  $B: WARN — worktree at $actual instead of expected $WT"
            echo "         (using existing path; submit_all_branches will too)"
            WT="$actual"
        fi
    fi

    if [ ! -d "$WT" ]; then
        echo "  $B: creating worktree at $WT..."
        if ! git -C "$REPO_ROOT" worktree add "$WT" "$B" 2>&1 | sed 's/^/         /'; then
            echo "  $B: ERROR — git worktree add failed" >&2
            fail_branches="$fail_branches $B"
            continue
        fi
    fi

    echo "  $B: git pull --ff-only origin $B..."
    if ! git -C "$WT" pull --ff-only origin "$B" 2>&1 | sed 's/^/         /'; then
        echo "  $B: WARN — pull --ff-only failed; investigate manually"
        fail_branches="$fail_branches $B"
        continue
    fi
    echo "  $B: HEAD = $(git -C "$WT" rev-parse --short HEAD)"
done

# ---- Step 3: clean __pycache__ across all worktrees -------------------------
echo ""
echo "[__pycache__ cleanup]"
for B in $BRANCHES; do
    WT="$PARENT_DIR/ts-prediction-$B"
    [ -d "$WT" ] || WT=$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null \
        | awk -v b="refs/heads/$B" '
            /^worktree / { wt = substr($0, 10) }
            $0 == "branch " b { print wt; exit }
        ')
    [ -n "$WT" ] && [ -d "$WT" ] || continue
    n=$(find "$WT" -type d -name __pycache__ 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  $B: purging $n __pycache__ dir(s)"
        find "$WT" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
    fi
done

# ---- Step 4: preflight check ------------------------------------------------
echo ""
echo "[preflight check]"
if [ ! -f "$REPO_ROOT/tools/preflight_check.sh" ]; then
    echo "ERROR: tools/preflight_check.sh not found at $REPO_ROOT/tools/" >&2
    echo "       (did the cb-AB pull actually include the new tools/ dir?)" >&2
    exit 1
fi

if BRANCHES="$BRANCHES" bash "$REPO_ROOT/tools/preflight_check.sh" --quick; then
    echo ""
    echo "==========================================="
    echo "All preflight checks passed."
    echo ""
    echo "Next steps on this gate node:"
    echo "  bash submit_all_branches.sh           # submit all 10 branches"
    echo "  squeue -u \$USER                       # confirm jobs queued"
    echo "  pestat                                # see node load"
    echo "==========================================="
    [ -n "$fail_branches" ] && {
        echo "NOTE: pull failed for:$fail_branches"
        echo "      preflight passed only because the existing checked-out"
        echo "      code was already up to date. Investigate the failed"
        echo "      pulls before relying on those branches' results."
        exit 1
    }
    exit 0
else
    rc=$?
    echo ""
    echo "==========================================="
    echo "Preflight FAILED — DO NOT submit jobs yet."
    echo "Each FAIL above maps to a contamination-class bug documented in"
    echo "the cross-branch contamination playbook."
    echo "==========================================="
    exit $rc
fi

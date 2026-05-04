#!/bin/bash
# ===========================================================================
# preflight_check.sh — diagnose cross-branch contamination risks BEFORE sbatch
# ---------------------------------------------------------------------------
# Run this from the cb-AB main repo before submitting any cb-* training run.
# It catches the failure modes seen in the 612171-612180 batch:
#
#   1. branch checked out in main repo (volatile HEAD → wrong code at runtime)
#   2. worktree at unexpected path (submit_all_branches looks at $PARENT_DIR)
#   3. worktree HEAD doesn't match branch name (manual `git checkout` drift)
#   4. stale __pycache__ in worktree (old branch's .pyc shadows current code)
#   5. logs/checkpoint/results not symlinked into the central output tree
#   6. run_*_slurm.sh missing branch-specific activation flag (cb-B/BC/BCD)
#   7. run_*_slurm.sh missing the explicit --csv fix for plot_metrics.py
#   8. submit_all_branches.sh missing the main-repo guard
#
# Usage:
#   bash tools/preflight_check.sh                      # check default branches
#   BRANCHES="cb-A cb-B cb-C" bash tools/preflight_check.sh
#   bash tools/preflight_check.sh --quick              # static checks only
#   bash tools/preflight_check.sh --smoke              # + 1-epoch smoke test
#
# Exit code: 0 if every branch passes, non-zero otherwise. Designed to be
# run as a final gate before `bash submit_all_branches.sh`.
# ===========================================================================
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null) \
    || { echo "ERROR: $SCRIPT_DIR is not in a git repo" >&2; exit 2; }
PARENT_DIR=$(dirname "$REPO_ROOT")

BRANCHES=${BRANCHES:-"reactot-halo8 cb-A cb-AB cb-ABD cb-B cb-BC cb-BCD cb-C cb-CD cb-D"}
WRAPPERS=${WRAPPERS:-"mix t1x"}
DO_SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) DO_SMOKE=1 ;;
        --quick) DO_SMOKE=0 ;;
        -h|--help)
            sed -n '2,/^# =====*$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "WARN: unknown argument '$arg'" >&2 ;;
    esac
done

# ---- Expected branch-flag table (kept in sync with run_*_slurm.sh) --------
expected_flag() {
    case "$1" in
        cb-B)   echo "--element-aware-weights" ;;
        cb-BC)  echo "--weighting-scheme BC" ;;
        cb-BCD) echo "--prior-scheme BC --learn-importance" ;;
        *)      echo "" ;;  # all other branches self-activate via defaults
    esac
}

# ---- Reporting helpers ----------------------------------------------------
total_pass=0
total_fail=0
per_branch_fail=""

color_ok="\033[32m"
color_warn="\033[33m"
color_err="\033[31m"
color_off="\033[0m"
[ -t 1 ] || { color_ok=""; color_warn=""; color_err=""; color_off=""; }

ok()   { printf "  ${color_ok}OK${color_off}    %s\n" "$1"; }
warn() { printf "  ${color_warn}WARN${color_off}  %s\n" "$1"; }
fail() { printf "  ${color_err}FAIL${color_off}  %s\n" "$1"; total_fail=$((total_fail+1)); branch_failed=1; }

# ---- Repo-wide static checks (run once, not per branch) -------------------
echo "==========================================="
echo "preflight_check.sh"
echo "REPO_ROOT  : $REPO_ROOT"
echo "PARENT_DIR : $PARENT_DIR"
echo "BRANCHES   : $BRANCHES"
echo "WRAPPERS   : $WRAPPERS"
echo "SMOKE      : $DO_SMOKE"
echo "==========================================="
echo ""
echo "=== Repo-wide checks ==="

branch_failed=0
# Check submit_all_branches.sh has the main-repo guard.
if [ -f "$REPO_ROOT/submit_all_branches.sh" ]; then
    if grep -q 'is checked out in main repo' "$REPO_ROOT/submit_all_branches.sh"; then
        ok "submit_all_branches.sh has main-repo guard"
    else
        fail "submit_all_branches.sh MISSING main-repo guard — cb-C / 612178-style contamination possible"
    fi
else
    warn "submit_all_branches.sh not found at $REPO_ROOT (only needed on submit host)"
fi
# Check plot_metrics.py supports --csv (it always has, but verify).
if [ -f "$REPO_ROOT/plot_metrics.py" ]; then
    if grep -q -- "--csv" "$REPO_ROOT/plot_metrics.py"; then
        ok "plot_metrics.py supports --csv"
    else
        fail "plot_metrics.py MISSING --csv argument — cannot pin CSV path explicitly"
    fi
else
    fail "plot_metrics.py not found at $REPO_ROOT"
fi
[ "$branch_failed" -eq 0 ] && total_pass=$((total_pass+1)) || per_branch_fail="$per_branch_fail repo"
echo ""

# ---- Per-branch checks ----------------------------------------------------
for B in $BRANCHES; do
    echo "=== $B ==="
    branch_failed=0

    # 1. Worktree exists at expected path (cb-AB exception: main repo).
    if [ "$B" = "cb-AB" ]; then
        WT="$REPO_ROOT"
        ok "cb-AB lives in main repo (by convention) at $WT"
    else
        WT="$PARENT_DIR/ts-prediction-$B"
        if [ ! -d "$WT" ]; then
            # Discover where the branch actually lives.
            actual=$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null \
                | awk -v b="refs/heads/$B" '
                    /^worktree / { wt = substr($0, 10) }
                    $0 == "branch " b { print wt; exit }
                ')
            if [ -n "$actual" ]; then
                if [ "$(cd "$actual" && pwd -P)" = "$(cd "$REPO_ROOT" && pwd -P)" ]; then
                    fail "$B is checked out in main repo (HEAD volatile, contamination risk) — see submit_all_branches.sh main-repo guard message for fix"
                    # Use main repo path for remaining checks; submit_all_branches will refuse anyway.
                    WT="$actual"
                else
                    # Fall back to actual path so the remaining checks run.
                    # This is fine for laptop dev (worktrees at .claude/worktrees/);
                    # cluster (where this script matters) will use $PARENT_DIR/ts-prediction-$B.
                    warn "$B worktree is at $actual (cluster expects $PARENT_DIR/ts-prediction-$B) — running remaining checks against $actual"
                    WT="$actual"
                fi
            else
                fail "$B has no registered worktree — submit_all_branches.sh will need to create one"
                continue
            fi
        else
            ok "worktree at $WT"
        fi
    fi

    # 2. Worktree HEAD matches branch name.
    head=$(git -C "$WT" rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ "$head" = "$B" ]; then
        ok "git HEAD = $B"
    else
        fail "git HEAD = $head (expected $B) — manual checkout drift; run: git -C $WT switch $B"
    fi

    # 3. Stale __pycache__ check.
    pyc_count=$(find "$WT" -type d -name __pycache__ 2>/dev/null | wc -l)
    if [ "$pyc_count" -eq 0 ]; then
        ok "no __pycache__ directories"
    else
        warn "$pyc_count __pycache__ dir(s) — slurm script will purge at job start, but bench-test runs from this worktree might import stale .pyc"
    fi

    # 4. logs/checkpoint/results are symlinks (or are the central output dir).
    output_dir=${OUTPUT_DIR:-"$REPO_ROOT"}
    output_real=$(cd "$output_dir" 2>/dev/null && pwd -P)
    wt_real=$(cd "$WT" 2>/dev/null && pwd -P)
    for sub in logs checkpoint results; do
        link="$WT/$sub"
        if [ "$wt_real" = "$output_real" ]; then
            # main repo: must be a real dir (the central output).
            if [ -d "$link" ] && ! [ -L "$link" ]; then
                ok "$sub/ is the central output dir"
            elif [ ! -e "$link" ]; then
                warn "$sub/ does not exist yet (submit_all_branches.sh will mkdir it)"
            else
                fail "$sub/ exists but is not a directory"
            fi
        else
            # Per-branch worktree: must be a symlink into output_dir.
            if [ -L "$link" ]; then
                target=$(readlink "$link" 2>/dev/null)
                if [ "$(cd "$(dirname "$link")/$(dirname "$target")" 2>/dev/null && pwd -P)/$(basename "$target")" = "$output_real/$sub" ] \
                        || [ "$target" = "$output_real/$sub" ]; then
                    ok "$sub/ symlink → central output"
                else
                    warn "$sub/ symlinks to $target (expected $output_real/$sub)"
                fi
            elif [ ! -e "$link" ]; then
                warn "$sub/ does not exist yet (submit_all_branches.sh will create the symlink)"
            else
                # Real dir in a per-branch worktree means output won't centralize.
                fail "$sub/ is a real directory in a per-branch worktree — output will NOT centralize. Run submit_all_branches.sh which calls ensure_output_layout to migrate it."
            fi
        fi
    done

    # 5. Branch-specific activation flag is wired in run_*_slurm.sh.
    expected=$(expected_flag "$B")
    for w in $WRAPPERS; do
        slurm="$WT/run_${w}_slurm.sh"
        if [ ! -f "$slurm" ]; then
            warn "run_${w}_slurm.sh not found in worktree (skip if this wrapper isn't used for $B)"
            continue
        fi
        if [ -n "$expected" ]; then
            # Look for the expected flag string anywhere in the file.
            if grep -F -q -- "$expected" "$slurm"; then
                ok "run_${w}_slurm.sh wires '$expected' for $B"
            else
                fail "run_${w}_slurm.sh MISSING activation flag '$expected' for $B — silent-collapse contamination per cb-B/BC/BCD playbook"
            fi
        fi
        # 6. plot_metrics call must pass --csv explicitly (the 612171-612180 fix).
        if grep -q "plot_metrics.py --csv" "$slurm"; then
            ok "run_${w}_slurm.sh pins plot_metrics.py to explicit --csv"
        else
            fail "run_${w}_slurm.sh calls plot_metrics.py without --csv — concurrent jobs will pick each other's CSV (612171-612180 incident)"
        fi
    done

    # 7. Optional smoke test: 1-epoch CPU run.
    if [ "$DO_SMOKE" = "1" ]; then
        echo "  (smoke test: 1 epoch CPU run, ~30s)"
        smoke_log=$(mktemp)
        (
            cd "$WT" || exit 1
            EXPERIMENT_ID="$B" RUN_NAME="preflight-${B}-$$" \
                python -u -m reactot.trainer.train_rpsb_ts1x \
                    --smoke-test --no-wandb --cpu \
                    >"$smoke_log" 2>&1
        )
        smoke_rc=$?
        if [ "$smoke_rc" -eq 0 ]; then
            ok "smoke test passed"
        else
            fail "smoke test FAILED (rc=$smoke_rc) — last 20 lines:"
            tail -20 "$smoke_log" | sed 's/^/        /'
        fi
        rm -f "$smoke_log"
    fi

    if [ "$branch_failed" -eq 0 ]; then
        total_pass=$((total_pass+1))
    else
        per_branch_fail="$per_branch_fail $B"
    fi
    echo ""
done

# ---- Summary --------------------------------------------------------------
echo "==========================================="
echo "Summary: $total_pass passed, $total_fail check(s) failed"
if [ "$total_fail" -gt 0 ]; then
    echo "Branches with failures:$per_branch_fail"
    echo ""
    echo "Do NOT submit training jobs until every branch passes. Each FAIL"
    echo "above corresponds to a known contamination mechanism documented"
    echo "in memory/feedback_branch_contamination.md."
    exit 1
fi
echo "All preflight checks passed — safe to run submit_all_branches.sh"
exit 0

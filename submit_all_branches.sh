#!/bin/bash
# ===========================================================================
# submit_all_branches.sh — submit one SLURM training job per experimental branch
# ---------------------------------------------------------------------------
# Each experimental branch (cb-*, rp-*, reactot-halo8) implements a different
# idea. To run them in parallel without cross-branch source contamination
# (see reactot/trainer/train_rpsb_ts1x.py::_git_branch_tag, which refuses to
# train if EXPERIMENT_ID disagrees with the live `git rev-parse HEAD` of the
# job's cwd), every job MUST run inside its own checked-out working
# directory — a `git worktree`.
#
# Default behavior:
#   1. For each branch in BRANCHES: locate an existing worktree (anywhere)
#      OR create a new one at "../ts-prediction-<B>"; pull origin/<B>;
#      set up output layout (see below); submit run_${WRAPPER}_slurm.sh
#      with DATA_LIMIT=$DATA_LIMIT EXPERIMENT_ID=<B>.
#   2. Then, for each branch in EXTRA_T1X_BRANCHES (default
#      "reactot-halo8"), additionally submit run_t1x_slurm.sh — this is
#      the T1x-only baseline that pairs with the augmented mix dataset.
#
# Output centralization (the reason every job's logs/checkpoint/results
# show up in the main repo, not scattered across 16 worktrees):
#   The trainer's cwd MUST stay in its own worktree (so `git rev-parse`
#   matches EXPERIMENT_ID; otherwise _git_branch_tag raises
#   "Cross-branch contamination detected"). But the user wants all
#   output visible in one place. Solution: per-worktree logs/,
#   checkpoint/, results/ are symlinked into $OUTPUT_DIR (default the
#   main repo). The job's cwd stays in its worktree, but all writes —
#   SLURM's --output=logs/<wrapper>_%j.out, the trainer's
#   checkpoint/<project>/<run-name>/ tree, plot_metrics.py's results/,
#   and PL CSVLogger's logs/<run-name>/ — funnel through the symlinks
#   into one shared tree. RUN_NAME embeds branch+job-id+rand so no two
#   jobs collide on a directory or filename.
#
#   Override OUTPUT_DIR=/some/other/path to centralize elsewhere; the
#   directory is created automatically. If a worktree already has
#   logs/, checkpoint/, or results/ as a non-empty real directory
#   (typically from earlier runs when the slurm body's mkdir + SLURM's
#   own mix_<jobid>.{out,err} capture left behind real dirs), its
#   contents are moved into the central tree and the dir is replaced
#   with a symlink. Migration is per-entry: any entry that already
#   exists in the central tree (extremely unlikely thanks to RUN_NAME's
#   uniqueness) is left in place with a WARN.
#
# Why this exists — the simpler inline loop that this replaces:
#     for B in $BRANCHES; do
#       WT="../ts-prediction-$B"
#       [ -d "$WT" ] || git worktree add "$WT" "$B"
#       ( cd "$WT" && sbatch run_mix_slurm.sh )
#     done
# fails for three distinct reasons that this wrapper handles:
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
#     the slurm script body runs too late to fix it. We pre-create the
#     output layout in the worktree before sbatch.
#
#   - With per-worktree logs/checkpoint/results, output ends up scattered
#     across 16 directories, so only the branch checked out in the main
#     repo (the one the user `cd`s into) appears to be "working" in any
#     casual file-explorer view. Symlinking funnels everything into one
#     shared location while preserving the per-worktree cwd that the
#     contamination check requires.
#
# Usage:
#   bash submit_all_branches.sh                                 # mix (all) + t1x (halo8)
#   WRAPPER=t1x bash submit_all_branches.sh                     # t1x for the main pass
#   EXTRA_T1X_BRANCHES="" bash submit_all_branches.sh           # disable the extra pass
#   OUTPUT_DIR=/scratch/$USER/reactot-out bash submit_all_branches.sh
#   BRANCHES="cb-A cb-B" bash submit_all_branches.sh            # subset
#   DATA_LIMIT=300 bash submit_all_branches.sh                  # smaller subset
#   DRY_RUN=1 bash submit_all_branches.sh                       # show plan only
#
# Recommended SSH workflow on UBAI cluster (gate1/gate2):
#   ssh -i ~/.ssh/yeseo1ee.pem yeseo1ee@172.16.xxx.xxx          # IP from email
#   cd ~/projects/ts-structure-prediction
#   bash tools/cluster_setup.sh                                 # pull all branches + preflight
#   bash submit_all_branches.sh                                 # only if preflight OK
#   squeue -u $USER                                             # confirm jobs queued
#   tail -f logs/mix_<jobid>.out                                # watch progress
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

# All worktrees' logs/checkpoint/results funnel here via symlinks. Default
# to the main repo so output appears in the directory the user normally
# `cd`s into.
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT"}

echo "==========================================="
echo "submit_all_branches.sh"
echo "REPO_ROOT          : $REPO_ROOT"
echo "PARENT_DIR         : $PARENT_DIR"
echo "OUTPUT_DIR         : $OUTPUT_DIR"
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

# Make sure the central output tree exists before any worktree links into it.
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/checkpoint" "$OUTPUT_DIR/results"

submit_count=0
fail_count=0
failed_jobs=""

# ---- Locate or create worktree for branch ---------------------------------
# Prints worktree path on stdout. All diagnostics go to stderr so the
# caller can do `WT=$(ensure_worktree X)` cleanly. Returns 1 on failure.
ensure_worktree() {
    local B="$1"
    local existing wt existing_real repo_real

    existing=$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null \
        | awk -v b="refs/heads/$B" '
            /^worktree / { wt = substr($0, 10) }
            $0 == "branch " b { print wt; exit }
        ')

    if [ -n "$existing" ]; then
        # Refuse to use the main repo for any branch OTHER than cb-AB. The
        # main repo's HEAD is volatile (fixes / cherry-picks / cleanup land
        # here constantly), and every parallel SLURM job would import the
        # source from whatever HEAD points to at job-start time, NOT the
        # branch tag SLURM was queued with. This is the cb-C / job-612178
        # incident: main repo happened to be on cb-C at submit, so cb-C ran
        # from /gpfs/.../ts-structure-prediction/ (proven by the leftnet.py
        # warning path in mix_612178.err) instead of a dedicated worktree.
        # cb-AB is a documented exception: by convention it always lives in
        # the main repo (memory: "main repo at <repo> holds cb-AB").
        existing_real=$(cd "$existing" 2>/dev/null && pwd -P)
        repo_real=$(cd "$REPO_ROOT" 2>/dev/null && pwd -P)
        if [ -n "$existing_real" ] && [ "$existing_real" = "$repo_real" ] \
                && [ "$B" != "cb-AB" ]; then
            echo "ERROR: branch '$B' is checked out in main repo $REPO_ROOT" >&2
            echo "       Only cb-AB may live in the main repo (its HEAD is" >&2
            echo "       volatile and would contaminate parallel runs)." >&2
            echo "       Fix on the cluster:" >&2
            echo "         git -C $REPO_ROOT switch cb-AB" >&2
            echo "         git -C $REPO_ROOT worktree add $PARENT_DIR/ts-prediction-$B $B" >&2
            echo "       Then re-run this script." >&2
            return 1
        fi
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

# ---- Recursive directory merge --------------------------------------------
# Merge every entry from $src into $dst. Atomic mv when there's no name
# collision; on collision, recurse if both sides are real directories
# (the typical case is checkpoint/RPSB-FT-Schedule/ — the project name
# is shared across all branches but the run-names inside are unique by
# construction). Returns 0 if every entry was successfully relocated
# (so $src is empty), 1 otherwise.
recursive_merge() {
    local src="$1" dst="$2"
    local item base failed=0 moved=0

    mkdir -p "$dst"

    while IFS= read -r -d '' item; do
        base=$(basename "$item")

        if [ ! -e "$dst/$base" ]; then
            if mv "$item" "$dst/" 2>/dev/null; then
                moved=$((moved+1))
                continue
            fi
            echo "WARN: failed to move $item into $dst/" >&2
            failed=$((failed+1))
            continue
        fi

        # Collision. Mergeable iff both sides are real directories
        # (not symlinks, not files).
        if [ -d "$item" ] && [ -d "$dst/$base" ] \
                && ! [ -L "$item" ] && ! [ -L "$dst/$base" ]; then
            if recursive_merge "$item" "$dst/$base"; then
                rmdir "$item" 2>/dev/null
                moved=$((moved+1))
            else
                failed=$((failed+1))
            fi
            continue
        fi

        # File-vs-file or file-vs-dir — can't auto-merge.
        echo "WARN: $dst/$base already exists; leaving $item in place" >&2
        failed=$((failed+1))
    done < <(find "$src" -mindepth 1 -maxdepth 1 -print0 2>/dev/null)

    if [ "$moved" -gt 0 ]; then
        echo "INFO: merged $moved entry/entries from $src into $dst" >&2
    fi
    [ "$failed" -eq 0 ]
}

# ---- Set up the output layout for one worktree ----------------------------
# Makes $WT/logs, $WT/checkpoint, $WT/results resolve to the corresponding
# subdirs of $OUTPUT_DIR. Idempotent across re-runs.
#
# When $WT IS $OUTPUT_DIR (i.e. the branch is checked out in the main
# repo), no symlinks are created — the dirs we mkdir'd at script start
# already ARE the worktree's output dirs.
#
# When $WT/<name> exists as a real directory (typically left behind by
# earlier runs that did mkdir + SLURM's mix_<jobid>.{out,err} capture),
# its contents are recursively merged into $OUTPUT_DIR/<name> and the
# dir is replaced with a symlink. mv is atomic and preserves open file
# descriptors, so the merge is safe even while SLURM jobs are writing.
# Recursion handles the checkpoint/<project-name>/ collision: every
# branch's worktree has the same top-level project dir, but the
# run-names inside are unique (RUN_NAME embeds branch + slurm-job-id +
# rand8), so descending one level lets them merge cleanly.
ensure_output_layout() {
    local wt="$1"
    local name target link wt_real out_real

    wt_real=$(cd "$wt" 2>/dev/null && pwd -P)
    out_real=$(cd "$OUTPUT_DIR" 2>/dev/null && pwd -P)
    if [ -n "$wt_real" ] && [ "$wt_real" = "$out_real" ]; then
        return 0
    fi

    for name in logs checkpoint results; do
        target="$OUTPUT_DIR/$name"
        link="$wt/$name"

        # Defensive: if $target doesn't exist, the `mv $item $target/`
        # inside recursive_merge would rename rather than move into.
        mkdir -p "$target"

        if [ -L "$link" ]; then
            ln -sfn "$target" "$link"
            continue
        fi
        if [ ! -e "$link" ]; then
            ln -sfn "$target" "$link"
            continue
        fi
        if ! [ -d "$link" ]; then
            echo "WARN: $link exists and is not a directory; skipping" >&2
            continue
        fi

        if recursive_merge "$link" "$target"; then
            rmdir "$link" 2>/dev/null && ln -sfn "$target" "$link"
        else
            echo "WARN: $link could not be fully merged into $target;" >&2
            echo "      $link remains a real directory and this branch's" >&2
            echo "      NEW output will write there instead of $target." >&2
        fi
    done
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

    # Set up output layout BEFORE sbatch. SLURM resolves --output=
    # against cwd at submit time but opens the file at job-start time;
    # if logs/ doesn't exist (or is a broken symlink) by then, stdout/
    # stderr capture silently fails. After this call, $WT/logs etc.
    # either are real dirs (when $WT == $OUTPUT_DIR) or are symlinks to
    # the corresponding $OUTPUT_DIR subdirs.
    ensure_output_layout "$WT"

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
echo "Output  : $OUTPUT_DIR/{logs,checkpoint,results}"
if [ -n "$failed_jobs" ]; then
    echo "Failed jobs:$failed_jobs"
fi
echo "==========================================="

[ "$fail_count" -eq 0 ]

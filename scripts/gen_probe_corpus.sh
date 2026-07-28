#!/usr/bin/env bash
# Overnight probe-corpus generation — implements probe-corpus-spec.md §5.
#
# Run from the conda env that has Isaac Sim + this repo's deps (the env you've been
# running collect_clip.py in). Generation only — feature extraction + audits
# (spec §6) run in `amass2` on GPU the next morning, NOT here (different env).
#
#   bash scripts/gen_probe_corpus.sh            # R=30 rollouts/scenario (default)
#   R=45 bash scripts/gen_probe_corpus.sh       # push more groups if render is fast
#
# Dominant lever is GROUPS, not cameras: --n_cameras 1 turns the render budget into
# independent rollout groups (the probe's group-split CV unit). R=30 ⇒ ~270 groups
# vs today's 17. Disjoint --seed bands keep rollout-group IDs globally unique.
#
# NOTE: re-running REGENERATES each scenario's dir from scratch (collect_clip clears
# the target). To *add* groups later, generate into a fresh OUT or bump R and re-run.

set -u                      # undefined-var guard; deliberately NOT -e (continue past a crash)

REPO="$HOME/vis_emb/brainstorm"
RESULT="$HOME/data/probe_corpus"
R="${R:-30}"                # rollouts per scenario (override via env: R=45 bash ...)
SCN=(drop roll throw drop_on_platform settle push cluttered topple collide)

# depth stays uint16 npz (the decision); CoM uses the GATE-A-verified legacy
# authoring. To also attack |CoM|~size, append `--com_offset_abs 0.01 0.06` to the
# collect_clip line below AND re-confirm GATE A first (it's a new authoring path).
COM_FLAGS=()                # e.g. COM_FLAGS=(--com_offset_abs 0.01 0.06)

# --- force the Isaac conda env, regardless of how this script was launched ----
# (last time it silently ran under `base` → import errors. Don't depend on the
#  caller's PATH: activate here, then HARD-FAIL if python isn't from the env.)
ENV_NAME="env_isaaclab"
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || { echo "FATAL: conda.sh not at $CONDA_SH"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || { echo "FATAL: could not 'conda activate $ENV_NAME'"; exit 1; }
case "$(command -v python)" in
    *"/envs/$ENV_NAME/"*) : ;;                       # good — python is from the env
    *) echo "FATAL: python is '$(command -v python)', not from $ENV_NAME. Aborting."; exit 1 ;;
esac
echo "env OK: ${CONDA_DEFAULT_ENV:-?}  |  python: $(command -v python)"

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
LOGDIR="$RESULT/_logs"; mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER="$LOGDIR/run_$STAMP.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

log "=== probe-corpus gen | R=$R/scenario | scenarios: ${SCN[*]} ==="
log "result=$RESULT  logs=$LOGDIR"

# # --- GATE A: label trust-gate. Abort the night if physics labels are wrong. ----
# log "GATE A: verify_physics_labels ..."
# if ! python src/data_gen/verify_physics_labels.py >"$LOGDIR/gate_a_$STAMP.log" 2>&1; then
#     log "GATE A FAILED — see $LOGDIR/gate_a_$STAMP.log. Aborting before any spend."
#     exit 1
# fi
# log "GATE A passed."

# --- generation: one scenario at a time, fail-tolerant ------------------------
for i in "${!SCN[@]}"; do
    s="${SCN[$i]}"; seed=$(( i * 1000 ))
    log "--- $s : $R rollouts, seed band $seed ---"
    t0=$(date +%s)
    if python src/data_gen/collect_clip.py \
            --result "$RESULT/$s" --scenario "$s" \
            --n_clips "$R" --n_cameras 1 --seed "$seed" "${COM_FLAGS[@]}" \
            >"$LOGDIR/${s}_$STAMP.log" 2>&1; then
        n=$(find "$RESULT/$s" -name meta.json 2>/dev/null | wc -l)
        log "$s OK in $(( $(date +%s) - t0 ))s -> $n clips"
    else
        log "$s FAILED (see $LOGDIR/${s}_$STAMP.log) — continuing to next scenario"
    fi
done

# --- morning summary ----------------------------------------------------------
log "=== SUMMARY ==="
total=0
for s in "${SCN[@]}"; do
    n=$(find "$RESULT/$s" -name meta.json 2>/dev/null | wc -l)
    total=$(( total + n ))
    flag=""; [ "$n" -lt $(( R * 7 / 10 )) ] && flag="   <-- SHORT (<70% of $R) — visibility-gated; bump R"
    log "  $(printf '%-18s' "$s") $n clips$flag"
done
log "TOTAL: $total clips / ${#SCN[@]} scenarios"
log "Next (amass2, GPU): extract + coverage_audit + density_audit + replicate_gate_b (spec §6)"

# E1 — probe-corpus acceptance runbook (v2 corpus, 2026-07-04)

*Executable checklist for accepting `~/data/probe_corpus` before any fit work.
Operationalizes [probe-corpus-spec.md §6](../data-gen/probe-corpus-spec.md) for the
corpus actually on disk; the ACTIVE experiment it feeds is in
[../EXPERIMENTS.md](../EXPERIMENTS.md). Code: `src/probe/corpus_acceptance.py`
(new), `density_size_audit.py` (recursive discovery fix), `run_probe.py extract`,
`coverage_audit.py` (both unchanged). E2 (gate_b_v2 fit matrix) starts only after
this page's checks are green.*

## What we already know about this corpus (read first)

- **It is a three-batch mix.** June-25 run: total failure (0 clips — see
  `_logs/run_20260625_*.log`). July-3 evening run (R=30): partial — `settle`/`push`
  crashed, master log ends mid-`cluttered`. July-4 overnight bulk (~700 clips):
  **no master `run_*.log` exists for it** — provenance is reconstructed from file
  mtimes by the acceptance script.
- **Seed bands are disjoint per scenario** (band = scenario-index × 1000), and a
  spot-check found all rollout seeds globally unique → the group split is safe.
  The acceptance script re-verifies this corpus-wide (HARD FAIL if not).
- **`--com_offset_abs` was NOT used** (`gen_probe_corpus.sh` ran with
  `COM_FLAGS=()`), so the spec's "`|CoM|~size` lower than +0.34" bar does **not**
  apply — expect it near the legacy +0.34..0.62. Record it; the fix is probe-side
  (normalized CoM target) and belongs to E2, not to regeneration.
- Known blemishes the script will confirm: **1 clip with `validation.ok=False`**,
  ~2 clip dirs missing `meta.json` (dir-count vs meta-count gap), and heavy
  validation gating on `throw` (~25 kept of ~115 seeds) and `collide` (~34/~98).

## Run order (CPU steps first; GPU pass last)

All from the repo root, in the probe env (`amass2`). Steps 1–2 need no GPU.

```bash
# 1. acceptance gate (CPU, seconds) — structure, groups, batches, settled fraction
python src/probe/corpus_acceptance.py --data ~/data/probe_corpus \
    --result_dir results/probe/acceptance
# read results/probe/acceptance/report.md; if it says HARD FAIL — stop, report back.

# 1b. if exclude_clips.txt is non-empty, move those clips out of the corpus root
#     (extract walks everything under --data; there is no exclude flag):
mkdir -p ~/data/probe_corpus_rejected
while read c; do mv ~/data/probe_corpus/"$c" ~/data/probe_corpus_rejected/; done \
    < results/probe/acceptance/exclude_clips.txt

# 2. density ⟂ size audit (CPU, ~1 min) — the R8 invariant on the new corpus
python src/probe/density_size_audit.py --data ~/data/probe_corpus \
    --result_dir results/probe/density_audit_v2

# 3. feature extraction (GPU, the one expensive pass; ~760 clips ≈ 7× the old
#    110-clip pass — budget a few hours; resumable only by rerun, so run it last)
python src/probe/run_probe.py extract --data ~/data/probe_corpus \
    --result_path results/probe/probe_corpus_feats.npz --device cuda

# 4. coverage audit on the cache (CPU, seconds)
python src/probe/coverage_audit.py --features results/probe/probe_corpus_feats.npz \
    --result_dir results/probe/coverage_v2
```

Do **not** run `replicate_gate_b`/fits yet — that is E2, and it changes shape
based on what steps 1–4 report (learning-curve protocol + target variants).

## Acceptance bars (what green looks like)

| # | check | source | bar | on failure |
|---|---|---|---|---|
| 1 | group uniqueness | acceptance report | 0 duplicates (**HARD**) | stop; report the duplicate list |
| 2 | contract files | acceptance report | 0 missing-required (**HARD**) | move offenders out (1b) |
| 3 | config homogeneity | acceptance report | 1 distinct config across batches | stop; a res/fps split is a hidden domain confound |
| 4 | groups/scenario | acceptance report | every scenario ≥3; note <½-median ones | E2 proceeds; flag starved scenarios for the top-up run |
| 5 | settled fraction | acceptance + coverage | \|v\| settled < 28% | E2 must motion-gate the sampled frame (spec §4) |
| 6 | `density ~ size` | density_audit_v2 | \|r\| < 0.2, spread ≳ 25× | **non-retrofittable** — R8 broken at the generator; stop before E2 |
| 7 | `\|CoM\| ~ size` | density_audit_v2 | *record only* (knob was off) | informs the E2 normalized-CoM target, not regen |
| 8 | target ranges | coverage_v2 | no ⚠ STARVED target rows | starved target ⇒ its E2 row is uninterpretable |
| 9 | samples survive pooling | extract console | skipped-clips count ≪ total | if extract drops many clips, investigate before fitting |

## What to report back (the E2 go/no-go packet)

1. `results/probe/acceptance/report.md` — verdict block + per-scenario table.
2. `results/probe/density_audit_v2/report.md` — the R8 verdict line + `|CoM|~size`.
3. `results/probe/coverage_v2/report.md` — range flags + groups-per-scenario bar.
4. From extract's console: total samples, skipped clips, cache size.

With those four in hand we lock the E2 fit matrix (learning-curve G-sweep +
target variants: instantaneous-vs-clip-mean `v`, motion-gated frame, normalized
CoM) and the classical-CV reference arm.

## Known deviations from spec §5 (accepted, documented)

- Scenario balance is uneven (throw/collide validation-gated hard) — accepted for
  E2; a **top-up run** for the light scenarios is the follow-up data-gen item
  (needs proportionally higher `--n_clips`, see keep-rate estimates in the
  acceptance report — not a same-R rerun).
- abs-CoM knob off (see above) — CoM decorrelation moves to the probe side.
- The corpus mixes generation dates; acceptance check #3 is the guard that this
  is cosmetic (identical configs) rather than a hidden split.

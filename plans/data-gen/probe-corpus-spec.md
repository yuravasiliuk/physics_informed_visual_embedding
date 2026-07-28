# Probe-corpus generation spec — stratified factor plan & command

*Date: 2026-06-23. The concrete, locked data-gen plan for the next (overnight)
probe corpus, derived from the coverage audits. Operationalizes
[scene-variability-rationale.md](scene-variability-rationale.md) §4 into knobs,
ranges, and a runnable command. Companion to
[physics-diversification.md](physics-diversification.md) (the W1/W2 invariants this
honors). Tracked in [IMPLEMENTATION.md](../IMPLEMENTATION.md).*

---

## What the audits told us (and how each maps to a knob)

| Finding (evidence) | Lever | Status |
|---|---|---|
| **Group-starved & imbalanced** — 17 groups, 5/9 scenarios at 1 group (`out/probe/coverage/`) | **`n_cameras=1` + balanced rollouts/scenario** — convert render budget into independent groups | **real, dominant** |
| **`density ⟂ size` HOLDS** (r≈0, 25× spread; `out/probe/density_audit/`) | leave density sampling alone; **decode density, not total mass** | **locked invariant** |
| **`|CoM| ~ size` = +0.34..0.62** | abs-CoM knob (partial) **+** probe-side normalization (primary) | **partial — see §3** |
| **`|v| ~ |ω|` = +0.60** | *none* — init ω₀⊥v₀ already holds; this is **emergent contact physics**, not a sampler artifact | **caveat, not a fix** |
| **28% near-motionless samples** | rebalance toward dynamic scenarios **+** probe-side motion gating | **mixed** |

Two findings demoted on inspection — state this plainly for the supervisor:

- **`ω₀ ⟂ v₀` is not a data-gen lever.** `scenarios._sample_angular_velocity`
  already draws ω₀ independently of every scenario's speed draw. The +0.60
  correlation is *emergent* (faster impacts tumble more) — real physics. It cannot
  and should not be "sampled away"; it is an **interpretation caveat** (a probe
  scoring on v may borrow ω signal and vice-versa), recorded, not fixed.
- **`|CoM| ~ size` cannot be driven to zero in data-gen.** Max physical CoM offset
  is bounded by the body's extent, so some `|CoM|`-vs-size coupling is *intrinsic*,
  not a confound artifact. The abs-CoM knob (§3) reduces it in the mid-size band;
  the real fix is **probe-side** (decode the dimensionless offset, or use the
  existing size-controlled CoM-lift metric).

---

## 1. Locked factor plan

Stratified by scenario (one generation call each), **uniform rollouts per
scenario** so no regime is group-starved. Everything else is sampled per body as
today, except the two knobs in §2–§3.

| factor | value | rationale |
|---|---|---|
| scenarios | all 9 (`drop, roll, throw, drop_on_platform, settle, push, cluttered, topple, collide`) | full regime span; **`collide` no longer at 1 group** (mass-observable regime) |
| rollouts / scenario | **15** (first batch) → scale if the probe still climbs | ≈9× the current ~1.7 groups/scenario; order-of-magnitude from the learning curve (plateau ⇒ size up, then re-measure) |
| cameras / rollout | **1** (`--n_cameras 1`) | groups, not views — the dominant lever |
| density | unchanged, `(200, 8000)` kg/m³ | `density ⟂ size` verified; **do not regress** |
| CoM offset | **absolute**, `--com_offset_abs 0.01 0.06` m | partial decorrelation of `|CoM|~size` (§3) |
| n_bodies | unchanged `(3, 8)` | — |
| ω₀, v₀ | unchanged (already independent at init) | — |

**Total first batch:** 9 × 15 = **135 independent rollouts → ~135 clips** (vs the
current ~110 clips that collapse to 17 groups). Same clip-count order, ~8× the
*independent* groups — exactly the binding constraint the learning curve named.

All three knob choices are **non-retrofittable** (they change dynamics/pixels):
locked here *before* scale per the data-gen invariants.

---

## 2. Groups, not cameras  *(enabler shipped)*

`collect_clip.py` gained an additive `--n_cameras` override (mirrors `--scenario`,
via `dataclasses.replace`; default = `ClipConfig.n_cameras=7`, unchanged). Setting
`--n_cameras 1` renders one viewpoint per rollout, so every clip is an independent
group. The probe's group-split CV treats rollouts — not cameras — as exchangeable
units, so this is the cheapest way to raise the group count.

---

## 3. CoM ⟂ size — partial in data-gen, primary fix in the probe  *(enabler shipped)*

`config.com_offset_abs_range` (+ `collect_clip.py --com_offset_abs MIN MAX`) is a
new, additive, default-`None` knob. When set, the **primitive** CoM offset is drawn
as an absolute magnitude in metres in a random direction, clamped to
`com_offset_frac·half_extent` so it stays inside the body
([body_factory.py](../../src/data_gen/body_factory.py)). Default `None` keeps the
legacy `frac·half_extent` behaviour, so existing pipelines are untouched.

**What it does and does not do.** It decouples `|CoM|` from size in the size band
where the clamp does not bind; for the smallest bodies the clamp still ties offset
to extent (physics: the CoM cannot leave the body). Composite bodies are
unaffected — their CoM emerges from density slabs and scales geometrically with
size. So expect the corpus `|CoM|~size` to *drop*, not vanish.

**Primary fix is probe-side** (cheaper, fully effective, no regen): decode the
**dimensionless** CoM offset (offset / half-extent), which is size-free by
construction, or rely on the existing **CoM-lift-over-centroid** metric in
`run_probe.py` (already controls for the size/centroid baseline). Recommended:
ship the abs-CoM knob in the corpus *and* switch the probe target to normalized
CoM — belt and suspenders.

---

## 4. Settled-frame mass

28% of samples are near-motionless because slow scenarios (`settle`, `cluttered`)
come to rest before the probe's center frame. Two levers:

- **Data-gen:** the uniform scenario mix already lifts the dynamic regimes' share
  vs the current topple/push-heavy corpus; no extra knob needed for the first batch.
- **Probe-side (recommended):** gate or motion-weight the sampled frame — pick the
  frame of peak body motion instead of the fixed center frame, or down-weight
  `|v| < 0.05` samples in the fit. This is a `dataset.py`/extractor change, tracked
  with the probe-setup fixes, not here.

---

## 5. Generation command

Run under Isaac (hand to the user — never run here). Each scenario gets a disjoint
seed base so rollout seeds are globally unique (distinct groups):

```bash
ISAAC="${ISAAC_SIM}/python.sh"
OUT=~/data/probe_corpus
R=15                      # rollouts per scenario (first batch)
SCN=(drop roll throw drop_on_platform settle push cluttered topple collide)

for i in "${!SCN[@]}"; do
  s="${SCN[$i]}"
  "$ISAAC" src/data_gen/collect_clip.py \
    --result "$OUT/$s" \
    --scenario "$s" \
    --n_clips "$R" \
    --n_cameras 1 \
    --com_offset_abs 0.01 0.06 \
    --seed $(( i * 1000 ))      # disjoint seed bands → unique rollout seeds
done
```

Label trust-gate first (GATE A discipline): before the batch, re-run
`verify_physics_labels.py` so the abs-CoM authoring still passes the closed-form
CoM check at the production camera.

---

## 6. Post-generation verification (close the loop)

After generation, extract features and **re-run both audits** on the new corpus;
acceptance criteria:

```bash
# 0. acceptance gate first (added with the v2 corpus): structure / groups /
#    batches / settled — see probe/e1-corpus-acceptance-runbook.md
python src/probe/corpus_acceptance.py --data ~/data/probe_corpus \
    --result_dir results/probe/acceptance
# 1. features (validated extractor)
python src/probe/run_probe.py extract --data ~/data/probe_corpus \
    --result_path results/probe/probe_corpus_feats.npz --device cuda
# 2. coverage — expect ~15 groups/scenario, |CoM|~bbox reduced, settled mass down
python src/probe/coverage_audit.py --features results/probe/probe_corpus_feats.npz \
    --result_dir results/probe/coverage_v2
# 3. density invariant — expect density~size still ~0, 25x spread preserved
python src/probe/density_size_audit.py --data ~/data/probe_corpus \
    --result_dir results/probe/density_audit_v2
# 4. re-measure the learning curve / probe on the bigger, balanced corpus
python src/probe/replicate_gate_b.py --features results/probe/probe_corpus_feats.npz \
    --result_dir results/probe/gate_b_v2
```

| check | expected | meaning if it fails |
|---|---|---|
| groups/scenario | ≈ 15 each | a regime still dropped clips to gating → boost its R |
| `density ~ size` | \|r\| < 0.2 | abs-CoM change accidentally touched density (it should not) |
| `|CoM| ~ size` | lower than +0.34 | clamp binding too often → widen size range or lower MAX |
| settled fraction | < 28% | scenario mix still rest-heavy → add motion gating (§4) |
| v-control R² | climbing past +0.05 | the corpus was the limiter — scale R further |

---

## Depth storage — keep uint16, move the size win into the codec

uint8 depth is rejected: it is 256× coarser than the current uint16 (per-clip
min–max), and because the scene depth span is dominated by camera standoff, the
small foreground bodies collapse to ~8 levels — their curvature/contact geometry
is destroyed. So **precision stays uint16**; storage is a *codec* question, now
encapsulated in [`depth_map_factory.py`](../../src/data_gen/depth_map_factory.py)
(`DepthMapFactory`) so codecs can be A/B-tested without touching capture.

Backends (selected via `ClipConfig.depth_format` or `collect_clip.py --depth_format`):

| backend | format | precision | notes |
|---|---|---|---|
| `npz` (default) | `depth.npz` (zlib) | bit-exact | legacy schema, unchanged |
| `ffv1` | `.mkv` gray16le | **bit-exact** | lossless 16-bit video, frame-scrubable |
| `hevc12` | `.mkv` gray12le | lossy, sub-mm | drops low 4 bits (≤ 15/65535 of span ≈ 0.5 mm); smallest |

Every encode writes a self-describing `depth_meta.json` sidecar (backend, d_min,
d_max, shape), so `DepthMapFactory.decode(clip_dir)` needs no prior knowledge;
legacy sidecar-less clips fall back to `depth.npz`. **Read-back uses the factory /
an ffmpeg pipe, never `cv2.VideoCapture`** (which is 8-bit and would silently
truncate).

**DECISION (2026-06-24, measured — `depth_format = "npz"`, locked for scale).**
Benchmarked on a real clip (`clip_00000`, 64×1024², 40 m span):

| option | size | precision |
|---|---|---|
| **`npz` (chosen)** | **11.09 MB** | bit-exact |
| `ffv1` native gray16 | 10.58 MB | bit-exact (only ~5% smaller — not worth the ffmpeg dep) |
| byte-split gray8 video | 25.65 MB | bit-exact but **2.4× bigger** (low byte is noise → won't compress) |
| `hevc12` | 0.48 MB | lossy, mean ~0.011%/px (≈4.6 mm here, ≤9 mm max) |

Lossless video only *ties* npz, so npz wins on simplicity (native numpy random
access, no ffmpeg dependency, no container footguns). The only real space lever is
lossy `hevc12` (~23×) — kept available for if/when storage bites and ~cm error is
acceptable, but **not** used for the probe corpus. The factory stays as the
encapsulation so this can be revisited without touching capture.

## Changes shipped with this spec (all additive, default-unchanged)

- `config.py` — `com_offset_abs_range: tuple = None`, `depth_format: str = "npz"`.
- `body_factory.py` — primitive CoM branch on `com_offset_abs_range`.
- `collect_clip.py` — `--n_cameras`, `--com_offset_abs`, `--depth_format` overrides.
- `depth_map_factory.py` (new) — pluggable depth (de)serialization; `clip_capture.py`
  delegates depth writing to it (`quantize_depth_uint16` re-exported, single source
  of truth); `collect_clip`/`collect_pairs` pass `config.depth_format`.
- `src/probe/coverage_audit.py`, `src/probe/density_size_audit.py` — the audits.

Defaults reproduce the old behaviour exactly; the new corpus is opted in entirely
through the §5 command flags.

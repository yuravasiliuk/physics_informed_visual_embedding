# Scene & Dynamics Variability — Rationale and Coverage Audit

*Date: 2026-06-23. Clarification note for research interns and the supervisor.
Bridges the probe results and the next data-gen run: it explains **why** we add
variability (and of what kind), then confronts that intent against an **empirical
coverage audit** of the realized corpus. Companion to
[physics-diversification.md](physics-diversification.md) (the locked W1–W5
decisions this re-checks), [scene-diversification.md](scene-diversification.md),
and the probe diagnostics under `out/probe/` (learning curve, diagnostics,
coverage). Tracked in [IMPLEMENTATION.md](../IMPLEMENTATION.md).*

---

## TL;DR

1. For this project, the payload of data-gen is **dynamical, not visual**. We add
   variability to make the *theories falsifiable* — i.e. so each physical factor
   under study actually **varies** and is **decorrelated** from cues the encoder
   could cheat on. Variability here buys **identifiability**, not realism.
2. An audit of the current 456-sample / 17-group probe corpus
   (`out/probe/coverage/`) says the frozen-probe plateau (~+0.05 test R²) is
   **not range starvation**. It is **(a) too few, badly-balanced independent
   rollout groups** and **(b) a residual size confound on CoM (+0.62)** — despite
   `physics-diversification.md` having *intended* to decouple it.
3. The core R8 test passed (§5): **`density ⟂ size` HOLDS** (r ≈ 0, 25× density
   range) — mass is not a deterministic function of visible size, so decode
   **density**, not total mass. The next (overnight) run must still generate
   **balanced independent rollouts** (several groups per scenario, not more cameras
   per rollout) under a **stratified sampler** that breaks the remaining
   confounds `CoM ⟂ size` (+0.62 cache / +0.34 meta) and `ω₀ ⟂ v₀` (+0.60). These
   are **non-retrofittable** knobs — settle them before scale.

---

## 1. Why variability — and why *not* mainly visual

V-JEPA is frozen. The early blocks already absorb a lot of the sim→real
appearance gap, so **texture/lighting/colour randomization is the cheap, secondary
axis.** The quantities the project actually wants the representation to carry —
velocity, angular velocity, centre-of-mass offset, mass/inertia — are *physical*,
and they only become decodable if the **dynamics** vary. A drop-only dataset can
have infinite visual variety and still teach nothing about mass (Galileo: in free
fall everything accelerates alike).

So the design principle is not "randomize for realism" but:

> **Vary each physical factor across its usable range, and decorrelate it from
> every cue the encoder could exploit instead.** A probe (or any downstream head)
> can only confirm a theory to the extent that the explanatory factor *varies* and
> a positive result is *attributable* to it rather than to a visible proxy.

Two corollaries that interns should internalize:

- **Range starvation is fatal.** If CoM offset never leaves ±5% of centre, no
  encoder can decode it — there is nothing to decode, and a "perfect" loss just
  means the head learned to output the constant. (This is the W2 defect
  `physics-diversification.md` already fixed at the *label* level; §3 checks
  whether it holds in the *realized* data.)
- **Confounds make positives meaningless.** If heavy objects are also big, a probe
  "decoding mass" is really decoding **size** (trivially visible). The fix is
  independent sampling, e.g. *same size, 3× density range*.

---

## 2. The variability axes (each tied to a theory it makes testable)

| Axis | What to vary | Theory it makes falsifiable |
|---|---|---|
| **Dynamical regime** | the `OBSERVABILITY_HINT` mix — ballistic / rolling / sliding / bouncing / **collision** / tipping / precession | *which equation-of-motion term is excited*. Mass is invisible in free fall, appears in collision; CoM appears in rolling/tipping. |
| **Inertial / latent** | density⊥size, CoM magnitude **and direction**, moment-of-inertia anisotropy (rod vs plate vs sphere, same mass) | R8 latent physics; the CoM "dichotomy". |
| **Contact-count ladder** | trajectories with a controlled **0, 1, 2, N** impacts | the world-line **degree/basis** claim (contact = C⁰ corner = slow polynomial convergence). To show "degree scales with #corners" you must *generate* a corner ladder. |
| **Damping regime** | a **viscous-drag / rolling-resistance** variant alongside Coulomb friction | Laplace/Prony vs Chebyshev: exponential-mode bases win on viscous decay, lose on dry friction — undiscriminable unless both regimes exist. |
| **Kinematic IC** | v₀, ω₀, drop height, launch angle, spin axis — **ω₀ independent of v₀** | velocity/ω recoverability; breaks the "fast = spinning" shortcut. |
| **Relational** | n_bodies 1/2/3, contact graph, interact vs independent | ego-line ⊕ others' world-lines. |

Camera/viewpoint is deliberately listed *last* and treated as an **observability**
control, not appearance diversity: a spin about the camera axis is invisible, so
viewpoint decides whether ω/CoM are *identifiable at all*. Prefer to control it
(≈1 camera per rollout) rather than spend the render budget multiplying
non-independent views — see §4.

---

## 3. Empirical reality check — the coverage audit

Source: `src/probe/coverage_audit.py` over `out/probe/sim_physics_feats.npz`
(the exact samples the probe sees, weight ≥ 0.5). Artifacts in
`out/probe/coverage/` (`report.md`, `histograms.png`, `audit.json`). This is the
step that tells us *which lever* the next run must pull, before we spend a night
of GPU. **456 samples / 17 groups.**

### 3.1 Range — is there anything to decode? *(adequate)*

| target | median | range | p95/p5 |
|---|---|---|---|
| \|v\| (m/s) | 0.571 | [0.003, 7.289] | 540 |
| \|ω\| (rad/s) | 2.156 | [0.000, 24.509] | 13141 |
| \|CoM\| (m) | 0.039 | [0.001, 0.195] | 22 |

All three targets span a usable dynamic range. **Range is not the binding
constraint** — one hypothesis for the plateau is eliminated. (bbox_diag is narrow,
p95/p5 ≈ 2.9, but it is a *confound axis*, not a decode target, so a narrow size
range is fine — even helpful.)

Caveat — **settled mass:** 28% of \|v\| and 27% of \|ω\| are near-zero (below
0.05 m/s, 0.10 rad/s). Over a quarter of the corpus is essentially motionless, so
the mean predicts those labels for free; SST is inflated by still bodies and the
probe is effectively graded on the moving ~70%.

### 3.2 Confound — can the probe cheat? *(yes, on CoM)*

Sample-level Pearson:

| pair | r | reading |
|---|---|---|
| \|v\| ~ bbox | −0.03 | clean |
| \|ω\| ~ bbox | −0.07 | clean |
| **\|CoM\| ~ bbox** | **+0.62** | **CoM offset tracks object size** |
| **\|v\| ~ \|ω\|** | **+0.60** | **fast bodies also spin fast** |

The `|CoM| ~ bbox = +0.62` is the headline. `physics-diversification.md` (W2)
*intended* offset CoM to be an independent latent — but the **realized** offset
magnitude still scales with body size, almost certainly because it is drawn as a
*fraction of body extent*. Consequence: a probe that appears to "decode CoM" may
be reading **silhouette size**, not physics — exactly the R8 trap the project
exists to avoid. The `|v|~|ω| = +0.60` is a second, milder shortcut (a probe can
borrow v-signal to predict ω and vice-versa).

> **Lesson for interns:** a *locked decision in a plan doc is an intent, not a
> guarantee.* Always audit the realized corpus. The label can be correct
> (CoM ≠ 0) and still be confounded (CoM ∝ size).

### 3.3 Coverage — independent groups per regime *(the dominant limiter)*

Groups = independent **physics rollouts** (the unit the probe's group-split CV
treats as one exchangeable sample). Cameras of the same rollout are **not**
independent.

| scenario | independent groups |
|---|---|
| topple | 5 |
| push | 3 |
| drop_on_platform | 2 |
| settle | 2 |
| cluttered / collide / drop / roll / throw | **1 each** |

Five of nine scenarios have a **single** independent group: they cannot survive a
train/test group split (they land entirely in train or entirely in test in every
fold). Group balance is lopsided overall (samples/group: min 6, median 25,
max 49). Body kind: 326 primitive samples / 17 groups vs 130 composite / 10
groups. Notably **`collide` has 1 group** — collision is the one regime where mass
ratio becomes observable, and it is currently the least represented.

This is the decisive read, and it confirms the learning curve
(`out/probe/learning_curve/`): the probe is **group-starved and group-imbalanced**,
not signal-starved.

---

## 4. Design rules for the next (overnight) run

Directly implied by §3. All four touch **non-retrofittable** knobs (they change
dynamics or pixels and cannot be patched into existing clips), so they must be
settled *before* the scale batch — per the data-gen invariants.

1. **Groups, not cameras.** Convert render budget into *independent rollouts*:
   ≈1 camera per rollout. Target **several independent groups per scenario**
   (no regime left at 1), with `collide` deliberately boosted.
2. **Break `CoM ⟂ size`.** Sample **absolute** CoM offset decoupled from bbox
   (not as a fraction of extent), so body size carries no information about CoM.
   Re-run the audit after generation; require `|CoM|~bbox ≈ 0`.
3. **Break `ω₀ ⟂ v₀`.** Initial spin sampled independently of initial translation
   (the realized corpus has `|v|~|ω| = +0.60`). Note `density ⟂ size` already
   holds (§5) and stays as a generator invariant — do not regress it; prefer
   **density** over total mass as the decode target.
4. **Trim settled frames.** Rebalance scenario/frame selection away from
   near-motionless samples so the probe is graded on bodies that actually move.

Prefer **stratified / Latin-hypercube sampling over i.i.d. randomization.** At a
few-hundred-sample scale, random draws leave coverage holes *and* manufacture
accidental correlations (this is plausibly how `|CoM|~bbox` crept in); a designed
sweep guarantees even coverage and orthogonality for free. Diversity and
group-count are orthogonal budgets — adding factors thins the sampling of all the
others, so fix the **factor list** first, then size the run so each factor is
adequately tiled.

---

## 5. The R8 core test — `density ⟂ size` *(now closed: HOLDS)*

`density/mass` are not in the feature cache (they live in clip `meta.json`), so the
core R8 test needed a separate pass: `src/probe/density_size_audit.py` over 79
unique dynamic bodies / 18 rollouts (camera-deduped). Artifacts in
`out/probe/density_audit/` (`report.md`, `scatter.png`, `audit.json`).

| pair | Pearson | Spearman | reading |
|---|---|---|---|
| **density ~ size** | **−0.00** | **+0.04** | **decoupled — R8 holds** |
| mass ~ size | +0.72 | — | geometric cube law (log–log slope **3.03**), not a density leak |
| \|CoM\| ~ size | +0.34 | — | real but milder than the cache's +0.62 |

**Verdict: R8 holds at the generator.** Density is sampled independently of visible
size (\|r\| ≈ 0) across a **25× density range** — W1 randomization is live, and mass
is *not* a deterministic function of visible volume. Three consequences:

- The "infer mass from dynamics, not appearance" premise is **sound at the source** —
  this is the one confound we feared most, and it is clean.
- **Decode density, not total mass.** `mass ~ size = +0.72` is the *expected*
  geometry (mass ∝ volume, slope ≈ 3) because density is free; total mass is
  therefore partly predictable from size, while **density is the truly latent,
  size-independent target.** Probes/heads should target density (or
  density-given-volume), not raw mass, to avoid crediting the encoder for geometry.
- **CoM ⟂ size still needs fixing**, but it is milder than the cache implied
  (+0.34 vs +0.62 — the cache value was partly inflated by pose / 2-D bbox
  projection). It remains on the §4 must-break list; `density ⟂ size` comes off it.

---

## Cross-references

- Intent this re-checks: [physics-diversification.md](physics-diversification.md)
  W1 (density⊥volume), W2 (offset CoM), W3 (observability scenarios), W5 (composite bodies).
- Probe evidence: `out/probe/coverage/` (target/confound/coverage audit),
  `out/probe/density_audit/` (R8 density⊥size, §5),
  `out/probe/learning_curve/` (group-count sizing),
  `out/probe/diagnostics/` (overfit vs no-signal).
- Audit scripts: `src/probe/coverage_audit.py`, `src/probe/density_size_audit.py`.
- Encoder/probe contract: [../probe/gate-b-probe.md](../probe/gate-b-probe.md),
  [../probe/vjepa-embedding-structure.md](../probe/vjepa-embedding-structure.md).
- Invariant being honored: latent physics must be inferable from geometry +
  dynamics only, never appearance (R8).

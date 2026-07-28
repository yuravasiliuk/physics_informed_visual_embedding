# Experiments — the hypothesis frontier

*The **third hub**, on a different axis from the other two:
[BRAINSTORM.md](BRAINSTORM.md) = **why** (architecture, risks R1–R11),
[IMPLEMENTATION.md](IMPLEMENTATION.md) = **what's built** (code status, gates),
this = **what we're learning** (the live tree of hypotheses → tests → results →
decisions). It is a **map, not territory**: every node is a one-liner that links
DOWN into the deep docs; it never duplicates them.*

**Last updated: 2026-07-18.**

## How this evolves (the five rules)

1. **One active line.** New ideas go to the **Parking lot** — they never hijack
   the active node. (This is the whole point: park, don't distract.)
2. **Results grow the tree, not speculation.** A `DONE` node's *decision* spawns
   its children; we don't pre-populate branches we're only imagining.
3. **Dead ends are `KILLED` with a reason, not deleted** (the Graveyard) — so
   nobody silently re-litigates them.
4. **Promotion is explicit.** Parked → active only when the active line resolves,
   chosen by stated value ranking.
5. **Nodes stay one-liners that link down.** Depth lives in the linked docs
   (the same recursion pattern the knowledge base already uses).

## Status legend

`ACTIVE` (≤1–2 at once) · `QUEUED` (next, defined) · `PARKED` (deferred, seeded) ·
`DONE` (resolved + decision) · `KILLED` (dead end + reason) · `BLOCKED` (waiting on a gate).

---

## ▶ YOU ARE HERE

**[ACTIVE] Corpus rebuild → physics-LoRA (Rung 1.5).**
**CoM IS ABANDONED (decision 2026-07-27)** — see the Graveyard. The oracle
proved it unobservable from perfect dynamics on the corpus; the marginal
eccroll certification (+0.020) is not worth the scenario budget it costs.
The active slot it occupied is now spent on targets that are *loudly*
observable: **restitution, friction, contact events**, plus the
twin-contrastive intervention signal.

**Next action, in order:**
1. **Material observability oracle** (no Isaac, no GPU) — reuse the
   `com_oracle` harness with restitution/friction as targets, on the
   EXISTING corpus. Settles `restitution_range` / `friction_range`
   **per scenario** before the rebuild bakes them in. This is the only
   pre-flight that can invalidate a freshly generated corpus.
2. **Collection wiring** — per-frame extrinsics + egomotion, contact
   logging, `collect_corpus` camera-mix flags
   ([data-gen/dynamic-camera.md](data-gen/dynamic-camera.md); trajectory
   layer already implemented + verified, 16/16 without Isaac).
3. **Pilot → verify-gate → scale**, then the LoRA line
   ([probe/physics-lora-adapter.md](probe/physics-lora-adapter.md)).

---

## Tree

**TRUNK — Does a frozen visual encoder linearly represent latent physics**
(enough to supervise a physics-informed embedding / sim→real bridge)?
Two hard gates ([IMPLEMENTATION.md](IMPLEMENTATION.md)): **GATE A** (dynamics
learnable from pixels) · **GATE B** (frozen-encoder probe → backbone).

```
GATE B — can frozen V-JEPA linearly decode v / ω / CoM?
│
├─ [DONE]    Plateau is group-starvation, not no-signal → CONFIRMED (co-limiter #1)
│     Result: v2 corpus (755 groups, E1 accepted) → gate_b_v2: v test −0.02 → +0.31±0.03
│             @ L32/40, ω +0.29 @ L30, smooth monotone layer curve.
│     Refs:    results/probe/{acceptance,density_audit_v2,coverage_v2,gate_b_v2}
│
├─ [DONE]    Probe-setup co-limiters → window misalignment CONFIRMED (co-limiter #2);
│            ★ CONTROL PASSES: v_clipmean R² 0.60±0.09 @ L32 (> 0.5 bar);
│            deltas: window +0.29, motion-gate +0.15; ω_clipmean 0.38 @ L30.
│     Decision: adopt WINDOW-ALIGNED (clip-mean) targets as Tier-1; kinematics
│             tap depth = layers ~29–32/40. ω below 0.5 linearly → arm 2b next.
│     Refs:    results/probe/fit_matrix, src/probe/{build_targets_v2,fit_matrix}.py
│
├─ [DONE]    CoM linear-null — artifact or absence? → NEITHER: **NOT OBSERVABLE
│            in the corpus** (encoder exonerated; null was uninformative)
│     Chain:  (a) scenario split: null UNIFORM (2a_v2.json) · (b) register
│             collapse RULED OUT (body↔frame cos falls 0.96→0.56 with depth)
│             · (c) V-JEPA 2.1 swap de-prioritized · (d) arm 2b: control 0.70
│             @ L32, ω 0.47 @ L30, CoM null survives MLP+concat · (e) ★ORACLE
│             (R9): GT trajectories → same head/folds give lift +0.001±0.002;
│             contact slices noise-negative. R6 VIOLATED IN PRACTICE.
│     Decision: GATE-B iteration-1 read locked — kinematics YES (v 0.60/0.70,
│             tap L28–32; ω 0.38/0.47, nonlinear gap +0.09 argues Rung-1);
│             CoM UNTESTABLE here (not "absent"); escape hatch NOT justified.
│             Fix data-gen before any further CoM probe/encoder spend.
│     Refs:    results/probe/{fit_matrix,register_collapse,arm_2b,com_oracle}
│
├─ [KILLED]  Make CoM dynamically observable (data-gen), oracle-gated
│     ABANDONED 2026-07-27 — CoM is dropped as a target entirely. Two pilots
│     + a mid-batch design could not make it loudly observable; the one
│     certified channel (eccroll, +0.020) buys ~11-13% of baseline and costs
│     a dedicated scenario budget. Spend that budget on restitution /
│     friction / contact events, which are observable by construction.
│     KEPT from this line: (1) the oracle METHOD — target-agnostic, no-GPU
│     scenario acceptance from state.json alone, now reused for materials;
│     (2) `com_camera` (body POSITION in camera frame) stays — it is the
│     kinematic channel, not the abandoned inertial offset.
│     Full history retained below for the no-re-litigation rule:
│     Insight: com_oracle is a NO-GPU acceptance test for scenario design —
│             pilot rollouts → oracle on state.json alone → scale only if lift.
│     So far: --frames 64 rerun: still null (subsampling exonerated). W3a NOT
│             the culprit (field-verified: topple rotates 70–160°); diagnosis =
│             SEPARATRIX RULE. W6 builders implemented → pilot #1 (25+25,
│             pilot_w6): oracle pooled +0.009 = MISLEADING composite (balance
│             −0.001, eccroll +0.020). Direct physics reads VALIDATE the
│             mechanism: balance tipping correlates with offset-toward-tip
│             (r=+0.34, beats geometry +0.28 in the contested band) but band
│             mis-centred (84% tipped); eccroll rest pose is a near-perfect
│             offset readout (cos +0.97) but only 15% reach rest by frame 64.
│             → RETUNED: balance band (−4.5,+0.5)°, kick ≤0.05; eccroll speeds
│             (0.4,1.2) + `--frame_count 128`; oracle got geometry features
│             (+ per-scenario PASS/fail verdicts, pooled demoted to reference).
│             Pilot #2 (pilot_w6_v2, oracle "SCALE nothing"): DECOMPOSED —
│             balance physics now RIGHT (38% tip; at the boundary geometry
│             decorrelates (−0.08) while offset still predicts (+0.25); toward
│             62% vs away 50%) but only ~12 contested bodies ⇒ the bar is
│             statistically unreachable at 25 rollouts. Eccroll rest fraction
│             UNCHANGED (14%) at half speed × 2× length ⇒ lossless rolling
│             (zero damping, no rolling resistance): rest-pose channel is
│             structurally rare; readout reframed as terminal OSCILLATION
│             about CoM-low (in-trajectory; confirmed cos +0.88 where rest).
│             Oracle means stay same-sign across independent pilots
│             (eccroll +0.020 / +0.016).
│             ★ (a) POOLED-ECCROLL ORACLE PASSES (v1+v2, n=96 bodies): MLP
│             +0.0161±0.0087, ridge +0.0199±0.0058 — positive in BOTH
│             hypothesis classes, clears the pre-registered bar. First
│             R6-CERTIFIED scenario: CoM offset demonstrably decodable from
│             its dynamics (~11-13% of baseline; ceiling likely higher —
│             generic MLP, no oscillation-phase features).
│     ★ (b0)  DONE (2026-07-06, results/probe/ecc46/): first INTERPRETABLE
│             encoder CoM read, on the R6-certified eccroll slice (96 samples
│             / 46 groups). Ridge control PASSES on the slice (v_clipmean
│             +0.558±0.163 @ L28 — corpus number replicates) → run valid.
│             LINEAR CoM: NULL — com_norm at the pre-registered band 28–32 is
│             +0.003/−0.007/−0.005 (best anywhere +0.008±0.011 @ L24;
│             mean−std<0 everywhere) vs oracle ceiling +0.020±0.006.
│             MLP: com_norm uniformly positive across all 7 slots (~+0.010,
│             best +0.013±0.010 @ L32, mean−std=+0.003) — BUT the MLP control
│             FAILED (v +0.328 vs 2a +0.558; 96 samples underfit the MLP), so
│             per arm_2b's own gate this row is a HINT, not evidence (slots
│             share folds — 7 positives are correlated, not independent).
│             Verdict: first informative encoder-side result — linear null +
│             nonlinear hint at ~½ the oracle ceiling; n=46 groups cannot
│             separate hint from noise ⇒ the mid-batch is now the decisive
│             test, not optional. Runbook: probe/b0-ecc46-runbook.md.
│     Next:
│             (b) MID-BATCH: balance ~200 rollouts @ --com_offset_frac 0.45
│             (its bar test — mechanism proven, n was the blocker) + eccroll
│             ~150 @ 128 frames (production allotment + firmer oracle stats)
│             → same bar on the mid-batch; (c) fold certified scenarios into
│             the probe corpus → the GATE-B CoM row becomes interpretable for
│             the FIRST time on those slices; (d) collide/throw top-up.
│     Refs:    results/probe/com_oracle{,_f64,_pilot,_pilot_v2,_ecc_all},
│             physics-diversification.md §W6
│
├─ [ACTIVE]  Corpus rebuild → observable targets → physics-LoRA (Rung 1.5)
│     Thesis: the CoM line failed on OBSERVABILITY, not on encoder capacity.
│             Replace it with targets whose observability is structural:
│             restitution (a bounce IS the readout), friction (slide/roll),
│             contact events (impulse ≈ Δ(m·v), closed-form verifiable).
│     Done:   camera trajectory layer (`camera_paths.py` +
│             `verify_camera_paths.py`, 16/16, no Isaac). ★ Gauge result:
│             the 4 unobservable DoF (3 translation + yaw about gravity) are
│             EXACTLY the motions leaving g_cam invariant → orbit/dolly give
│             zero tilt coverage, arc moves along the degeneracy, ONLY roll
│             breaks x≡0. Gravity tilt DEMOTED to enabling-signal/thermometer
│             (appearance-recoverable; cf. GeoCalib) — not the physics claim.
│     ★★(1)  MATERIAL ORACLE — RESTITUTION IS OBSERVABLE (2026-07-27).
│             The blocker was the WINDOW, not the parameter range. Same heads,
│             same folds, same labels; only the feature window changed:
│             generic 32-frame summary (391 dim) → ridge −0.003 / MLP −0.018;
│             EVENT-ANCHORED ±4 frames at impact (58 dim) → ridge +0.129±0.032
│             / MLP +0.217±0.037. Null → +0.22 with 7× FEWER features = a
│             misalignment problem, not a capacity one. Passes 8/9 scenarios
│             (only `collide` nulls, n=57).
│             ⇒ restitution_range (0.0-0.4) STAYS — one fewer non-retrofittable
│             knob; the earlier "widen the band" reading is RETRACTED.
│             ⇒ GATE-B's window lesson sharpens: window-aligned bought +0.29 for
│             velocity; impulsive quantities need EVENT-ALIGNED targets. A
│             clip-mean encoder probe cannot see restitution at any encoder
│             quality — the target FORM is load-bearing.
│             ⇒ contact logging doubly justified: exact impulses AND the event
│             timestamps that make event-anchored windows constructible.
│             ⚠ Method: the oracle was NOT an upper bound here (unlike for CoM,
│             where the offset shapes the whole trajectory). Any null must be
│             retested on the target's OWN event before being believed.
│             Dynamic friction: still null, but impact-anchored features are
│             restitution's event, not friction's → `--anchor slide` pending.
│             Static friction: DROP (slip/stick separatrix-only — the CoM trap).
│     (1a)    earlier diagnostic pass, kept because it localized the problem
│             (2026-07-27, diagnostics on the v2
│             corpus: 3770 bodies, 7257 bounce events). Blocker identified =
│             TEMPORAL UNDERSAMPLING, not absence. Coverage is fine (91.1% of
│             bodies bounce, 75% slide) but state is logged per RENDERED frame,
│             so gravity injects g·Δt = 0.327 m/s between samples while the
│             median predicted rebound e·|vz_in| = 0.178 m/s — BELOW the floor.
│             Only 33.8% of rebounds survive one frame ⇒ corr(e_est,e_true)
│             +0.081 pooled, but +0.278 above-floor (+0.319 gravity-corrected)
│             and +0.639 in `topple` (high-energy impacts). Fix order:
│             contact-impulse logging (exact, sampling-free — now ESSENTIAL,
│             not optional) > raise e·|vz_in| per scenario > 60 Hz state log
│             (halves the floor only). Decodability read (ridge/MLP) still
│             pending — needs torch/env_isaaclab.
│             Code: src/probe/material_oracle.py (reuses the com_oracle harness).
│     Scenarios (decided 2026-07-27): DROP `balance` + `eccentric_roll` — both
│             existed only for CoM, and both are weak for the new targets
│             (balance = few high-energy impacts; eccroll = lossless rolling,
│             which carries no friction information). Neither was ever in
│             `config.scenarios`, so the default mix is unchanged; builders are
│             marked DEPRECATED, not deleted (CLAUDE.md §Retire-don't-delete).
│             KEEP + PROMOTE `collide`: it is ~1.8% of the corpus (63/3511) yet
│             it is the ONLY scenario where mass/density RATIO is exactly
│             observable (m₁Δv₁ = −m₂Δv₂) — the structurally-observable latent
│             inertial target that CoM failed to be. Needs a bigger share AND a
│             third event anchor (`--anchor collision`, keyed on relative
│             approach velocity): the current impact detector sees only VERTICAL
│             sign flips, so horizontal body-body momentum exchange is invisible
│             to it — the likelier reason `collide` nulled, over n=57.
│     Built (2026-07-28, none run under Isaac yet):
│             camera_paths.py (orbit) + camera_motion.py (free 6-DoF: the orbit
│             model could not express pan/tilt/truck at all) + contact_log.py
│             (solver-exact impulses at PHYSICS rate — bypasses the render-rate
│             floor that destroyed restitution, and supplies the event
│             timestamps event-anchored targets are defined by) + collection
│             wiring (camera.json, contacts.json, schema v1-dyncam) + three
│             no-Isaac verifiers (camera paths 18/18, camera motion, contacts).
│             User is writing the 8 motion families.
│     Next:   (1b) run the full oracle under env_isaaclab for the ML read;
│             (2) collection wiring (per-frame extrinsics + egomotion +
│             contact logging); (3) pilot → verify-gate → scale; (4) LoRA.
│     Refs:   data-gen/dynamic-camera.md, probe/physics-lora-adapter.md,
│             probe/lora-implementation-notes.md
│
└─ [DONE]    (earlier decisions — see "Resolved" below)

GATE A — is dynamics learnable from raw pixels at all?
└─ [PARKED]  Overfit harness built (TinyClipEncoder + DynamicsModel; CPU smoke v R²→1.0),
             not run at scale.  Refs: src/gate_a/
```

## Parking lot (deferred branches, seeded to resume cold)

### [PARKED] Soft bodies — leverage deformation as *visible* physics
**Promotion condition:** after the rigid GATE-B result is in (it informs whether
this is an extension or the better main bet).

**The thesis (three threads converge — resurrect from here):**
1. **Deformation is a visible readout of internal physics.** Rigid latents
   (mass/CoM) are invisible by design (R8) — the probe is hard *because* the
   encoder must infer them indirectly. Softness inverts this: you *see* the strain;
   how much a body squishes under load *is* its stiffness. Tight appearance↔physics
   coupling rigid bodies can't have.
2. **It's in-distribution for V-JEPA.** V-JEPA trained on natural video (people,
   cloth, hands) — deformation, not clean rigid primitives. **Hypothesis:
   stiffness-from-deformation decodes *better* than CoM**, and the rigid plateau
   may partly be that we picked targets the encoder has little reason to represent.
3. **It generalizes the theory, doesn't fork it.** Rigid body = the
   **infinite-stiffness limit** of a soft body (all material points share one
   world-line + SO(3)). Stiffness is a *coupling parameter on the world-line
   bundle*; lower it and the material world-lines decouple → fluid. And the
   deformation modes are damped eigenmodes — exactly where the **Laplace/Prony
   exponential basis wins** (it lost on rigid Coulomb contact).

**Costs (why it's parked, not active):** heavy FEM/PBD sim (slow, finicky); the
**label contract evaporates** — `verify_physics_labels` is closed-form for 6-DoF
rigid state, there is no closed form for a deformation field, so a whole new GT
contract + audit suite is needed; it's a parallel data-gen track that multiplies
unknowns before the rigid probe is even proven.

**Cheapest de-risk (do this BEFORE any Isaac soft-body build):** probe frozen
V-JEPA on *existing* deformable video (a real bouncing-soft-ball clip, IntPhys
deformables) and check if deformation state is linearly *more* decodable than
rigid inertials. If that toy check lights up, the sim investment is evidence-backed.

### [PARKED] Rung 1.5 — physics-LoRA adapter on frozen V-JEPA (upper blocks, real-anchored)
**Promotion condition:** GATE-B iteration-2 read locked (post-W6 corpus) **and**
friction/restitution observability certified by the generalized oracle (its §4.1).

**Corpus rebuild (2026-07-27):** the probe corpus is being **regenerated from
scratch** with a dynamic-camera slice — [data-gen/dynamic-camera.md](data-gen/dynamic-camera.md)
(trajectory layer implemented + verified: `camera_paths.py`,
`verify_camera_paths.py`, 14/14 without Isaac). Blocking pre-flight, no Isaac and
no GPU: **material observability oracle** (restitution/friction from GT
trajectories on the EXISTING corpus) — `restitution_range` is dynamics-changing
and per-scenario, so it must be settled before the rebuild, or we repeat the CoM
mistake at corpus scale.

One-liner: LoRA on blocks 24–32 (tap-depth evidence L28–32; truncated at the
deepest tap), trained on
twin-contrastive intervention pairs (collect_pairs generalized to
friction/restitution) + sparse material/outcome heads, anchored to real video;
IntPhys2 surprise = drift canary; rank sweep = memorization dose-response.
Answers "can the encoder be cheaply *steered*" (≠ TRUNK's "does it already
encode"); candidate replacement for the Rung-2 two-stream escape hatch.
Compute is probe-scale via cut-layer activation caching. Full design:
[probe/physics-lora-adapter.md](probe/physics-lora-adapter.md). Dynamics-neutral
prerequisites (contact logging, twin driver, future-state targets) can ride
along with the next collection batch — see its §6.

### [PARKED] World-lines — Tier-2 (supervisor: after the probe)
GT encode→decode→spatiotemporal-plot baseline exists; rotation bug fixed. Fix the
batch of known bugs before extending — [IMPLEMENTATION.md §Known bugs](IMPLEMENTATION.md#known-bugs)
**B1–B4** (composite CoM field, visible-flag, composition-dependent gauge,
`to_gauge` quat) — then E0 tests + geodesic reporting. Deliverable order B
(representation contract) → C (E0) → D (ablation). Refs: `src/worldlines/`,
[critique/worldlines_supervisor_instructions.md](critique/worldlines_supervisor_instructions.md).

### [PARKED] IntPhys2 Option-A surprise — action-free masked-future-prediction
Pipeline built (`src/intphys/`), never run on GPU. Validates the encoder on a
real benchmark independent of the sim probe. Refs: [probe/intphys2-surprise.md](probe/intphys2-surprise.md).
**Correction (2026-07-27): a sibling implementation WAS run.**
`~/vis_emb/vjepa2/notebooks/intphys2_surprise.py` (untracked, 2026-06-21) is more
complete than `src/intphys/` — it carries the full IntPhys2 protocol
(Possible/Impossible pairing, pairwise accuracy, AUROC, per-condition breakdown)
— and `intphys2_surprise_scores.npz` beside it holds **1012 scores**. So a
frozen-encoder baseline plausibly already exists, which is exactly what the LoRA
**drift canary** needs. TODO: reconcile the two implementations (keep the
notebook's protocol?) and re-mark this node. Provenance of `src/intphys/surprise.py`
is verified-derived from the vjepa2 repo (loss = `app/vjepa/train.py:432,440-447`;
construction = `src/hub/backbones.py:89-131`); the *temporal causal* masking and
sliding-window aggregation are ours, and remain unvalidated on GPU.

## Resolved (DONE — decision recorded, feeds the tree)

- **[DONE] Density ⟂ size (R8 core) — HOLDS.** r≈0 (Pearson/Spearman), 25× density
  spread. → **decode density, not total mass** (mass~size +0.72 is the geometric
  cube law). Refs: `out/probe/density_audit/`, [scene-variability-rationale.md §5](data-gen/scene-variability-rationale.md).
- **[DONE] Depth storage — keep uint16 `.npz`.** Lossless video only ties npz;
  precision win of uint16 over uint8 is 256×. → `depth_format="npz"` (default).
  Refs: [probe-corpus-spec.md §Depth](data-gen/probe-corpus-spec.md), `src/data_gen/depth_map_factory.py`.
- **[DONE] Encoder path == reference notebook.** Repo-native ViT-g/384, bit-identical
  (max diff 0.0). → trust the frozen features; limits are data/probe, not encoder.
  Refs: `src/probe/encoders.py`, [probe/vjepa-embedding-structure.md](probe/vjepa-embedding-structure.md).
- **[DONE] Non-retrofittable knobs settled for scale.** 1 cam/rollout, balanced
  scenarios, depth=npz, density⟂size invariant, CoM-abs optional + GATE-A-gated.
  Refs: [probe-corpus-spec.md](data-gen/probe-corpus-spec.md).

## Graveyard (KILLED — do not re-litigate)

- **[KILLED] CoM / centre-of-mass offset as a target — ABANDONED 2026-07-27.**
  Not "unproven" — *measured*: the R9 oracle showed the offset is not decodable
  from **perfect GT dynamics** on the corpus (+0.001±0.002), so the encoder's
  null was uninformative and no probe-side fix could have helped. Two W6 pilots
  chased observability into a corner: `balance`'s mechanism validated but was
  statistically unreachable at pilot n; `eccentric_roll` certified at only
  +0.020 (~11–13% of baseline) because lossless rolling makes its rest-pose
  channel structurally rare. **Decision: the scenario budget buys more from
  restitution / friction / contact events, which are observable by
  construction.** Dropped: `com_offset`/`com_local` authoring, `com_norm`
  targets, `--com_offset_frac/_abs`, CoM-scale twins. **Kept:** the oracle
  *method* (target-agnostic no-GPU acceptance gate) and `com_camera` (body
  POSITION in camera frame — the kinematic channel, unrelated to the offset).

- **[KILLED] uint8 depth.** 256× coarser than uint16; foreground bodies collapse to
  ~8 levels (curvature/contact destroyed). Per-clip global normalization spends the
  budget on camera standoff.
- **[KILLED] Lossless depth video (FFV1 gray16).** Only ties npz (10.6 vs 11.1 MB on
  a real clip) — not worth the ffmpeg dependency / container footguns.
- **[KILLED] Byte-split depth video.** 2.4× *bigger* than npz (25.6 MB) — the low
  byte is noise and won't compress.
- **[KILLED] "39 groups clears R²=0.5" extrapolation.** The reliable region of the
  learning curve is a plateau, not a line; the slope was fit through noisy low-G transition.
- **[KILLED] Single-split "+0.33 v @ layer 27".** Overfitting artifact (1 test group);
  k-fold gave test −0.02.
- **[KILLED] ω₀ ⟂ v₀ as a data-gen lever.** Init ω₀ is already independent of v₀; the
  corpus +0.60 |v|~|ω| is emergent contact physics, not a samplable confound — an
  interpretation caveat, not a fix.

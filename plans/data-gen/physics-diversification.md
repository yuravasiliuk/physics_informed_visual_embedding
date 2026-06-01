# Physics Diversification & Verify Hardening

*Date: 2026-05-30. The data-gen fidelity pass that must land **before** any scale
generation (P1) and before the feasibility/architecture gates lean on physics
labels. Companion to [data_gen_critique0_analysis.md](../critique/data_gen_critique0_analysis.md)
(#4, #7) and [observability-weighted-loss.md](../theory/observability-weighted-loss.md)
(R6). Tracked operationally in [IMPLEMENTATION.md](../IMPLEMENTATION.md) P0 hardening.*

## The confound this fixes

Trajectory diversity is already decent (7 scenarios). The defect is one level
deeper: **every dynamic body is a uniform-density primitive with CoM at the prim
origin**, so the *latent physical identity* the project claims V-JEPA encodes
never varies independently of appearance.

- `body_factory.build_random_bodies` spawns only primitives and never sets
  density → `mass = 1000 · volume`, a deterministic function of visible shape+dims.
- `read_local_com` returns `[0,0,0]` for every primitive → the CoM-offset label is
  **identically zero across the entire dataset**.

Consequences, stated as failures we would otherwise discover too late:

| Head / question | Failure on current data |
|---|---|
| CoM-offset (R7, F-config) | label has zero variance → head learns "output 0", scores a deceptively perfect loss, proves nothing |
| Mass ⟂ size (Direction A, #7) | mass = volume → a positive probe just means "learned volume" |
| Inertia | analytic uniform tensor, fixed by shape+dims → no independent signal |

This is the precise mechanism by which sloppy data-gen yields mistaken
conclusions: the quantities under study carry no signal the encoder could be
credited (or faulted) for representing.

A second, paired defect: **even if we add offset CoM, the current scenarios rarely
*excite* it.** CoM is revealed by tumbling about an off-center point and by
contact; mass only by momentum exchange (collisions). Free-flight + flat ground
impact give weak observability, so a labeled-but-unobserved offset is still
unanswerable (the R6 "no-signal-in-data" half).

## Decisions (locked)

1. **CoM/mass mechanism — authored first, composite later.** Phase 1 authors
   `density` + `centerOfMass` (+ inertia) directly on the existing primitives:
   exact closed-form ground truth, cheap, reproducible, and appearance gives **no
   cue** — the purest test of "infer CoM from dynamics alone." Phase 2 adds
   composite glued multi-primitive bodies for realism, once the offset-CoM code
   path is trusted by verify.
2. **Observability scenarios — add tipping/toppling + inter-body collisions**, so
   the new latent properties are actually observable, not just labeled.

## Work plan (dependency order)

### W1 — Density randomization (decouple mass from volume) ✅ DONE
*Fixes #7. Smallest, unblocks "mass ⟂ size".*

- Add `density_range` to `ClipConfig` (e.g. `(200.0, 8000.0)` kg/m³ — styrofoam→steel).
- In `build_random_bodies`, sample a per-body density, apply via `UsdPhysics.MassAPI`
  (`CreateDensityAttr`), and **log `density` + resolved `mass` into `body_specs`**
  (also satisfies #8/#9 — mass once in specs, not only per-frame in `state.json`).
- Decouples mass from visible volume. Cannot be retrofitted (changes dynamics).

### W2 — Authored offset CoM on primitives (the key change) ✅ DONE
*Fixes R7. The CoM-offset label gains real variance.*
*Implemented in `body_factory.spawn_body`: per-body offset CoM (per-axis
fraction `ClipConfig.com_offset_frac=0.3` of the half-extent, bounded inside
the collision volume) authored with explicit mass + diagonal inertia via
`attach_mass_properties` (supersedes the W1 density-only path; explicit mass
takes precedence in PhysX → mass/CoM/inertia all exact). Helpers
`half_extents` + `inertia_diagonal` (uniform-solid centroidal tensor) added;
inertia closed forms validated against Monte-Carlo to <0.03% (capsule 0.01%).
Logged: `com_local_authored`, `inertia_diag_local`. `read_local_com` remains
the measured post-reset cross-check — asserted equal in W4. Compile-checked
only; not yet run under Isaac.*

- Sample a per-body CoM offset inside the shape's interior (a fraction of the
  half-extent, bounded so it stays physically inside the collision volume).
- Author it via `MassAPI.CreateCenterOfMassAttr`; **author mass + diagonal inertia
  explicitly too** so all three are known exactly rather than PhysX-recomputed —
  this is what makes the verify closed-form (see W4).
- Log `com_local_authored` in `body_specs` (exact GT), and keep `read_local_com`
  as the *measured* cross-check (they must agree post-reset).
- **Predict CoM as an offset from the centroid, and always score the lift over the
  "predict 0 / centroid" baseline** (R7 — guards against an absolute-position confound).
- Open question this forces us to resolve (see "Correctness risk" below):
  whether `get_world_pose()` returns the prim origin or the CoM for an offset-CoM
  body. W4 settles it; until then the `pos + R·com_local` reconstruction is
  unverified for the non-zero case.

### W3 — Observability scenarios ✅ DONE (topple fix in W3a)
*Fixes the R6 no-signal-in-data half. New builders in `scenarios.py` + registry.*
*Implemented: `build_topple` + `build_collide`, registered in `BUILDERS` and
`ClipConfig.scenarios`; per-scenario `observability_hint` tag injected in
`scenarios.build`. Per-body construction refactored into `body_factory.spawn_body`
(shared with `build_random_bodies`, RNG order preserved). `topple`'s CoM-face
signal is dormant until W2. Compile-checked only — not yet run under Isaac.*

#### W3a — `topple` produces near-zero motion (energy-barrier bug) ⬜ TO IMPLEMENT
*Found on inspection 2026-05-30: the `topple` scenario looks static.* The tip
kick `TOPPLE_TIP_ANGVEL_RANGE = (1.0, 2.5)` rad/s is **at or below the tipping
energy barrier**. For a typical tall body (footprint half ≈0.035 m, height half
≈0.175 m) the critical tilt is θ_crit ≈ 11°, so the CoM must rise Δh ≈ 3.5 mm to
clear the support edge — barrier ≈ `m·g·Δh ≈ 0.034·m` J. The kick supplies only
`½·I·ω² ≈ 0.005·m` at ω=1.0 and `≈0.032·m` at ω=2.5 — under the barrier — so the
body rocks and settles back upright instead of toppling.

**Fix (decided with user): lean + small kick.** Spawn each topple body already
tilted *just past* its critical angle so **gravity** does the toppling
(guaranteed visible every clip), with a small angular kick in the tip direction:

- Add an `orientation` (wxyz quaternion) parameter to `body_factory.make_body`
  and `spawn_body` (Isaac `Dynamic*` constructors already accept `orientation`).
- In `build_topple`: sample a tip azimuth φ; lean angle = θ_crit + margin (≈5°)
  about the horizontal axis ⟂ to the tip direction `(−sinφ, cosφ, 0)`. Place the
  body so its **lowest tilted vertex sits at ground+clearance** (shift center up
  by `Σ_k |R[2,k]|·half_extent_k`) — no t=0 penetration, no float. Apply a small
  angular kick `ω = mag·(−sinφ, cosφ, 0)` (continues the tip toward φ).
- With the authored offset CoM (W2), gravity-driven toppling lets the offset
  bias *which* face it lands on without the kick swamping the CoM signal.
- Log `lean_deg`, `tip_azimuth_rad` in `scenario_specs`.

- **`topple`** — a tall body (high aspect-ratio cuboid/cylinder) spawned upright or
  marginally stable; it tips and settles onto a face. With an authored offset CoM,
  *which* face it settles on is decided by the offset → CoM is strongly observable.
- **`collide`** — two (or more) dynamic bodies on intersecting paths (reuse the
  `push`/`throw` velocity machinery) so they impact. Momentum exchange makes the
  **mass ratio** observable — the only way mass becomes a learnable target.
- Keep the existing 7 trajectory scenarios. Tag each scenario in `scene_specs` with
  a coarse expected-observability hint (free-flight / contact-rich) so the
  observability-weighted loss (R6) and dataset-balancing can use it later.

### W4 — Verify hardening ✅ DONE — all 3 cases pass; 2 bugs found & fixed
*Fixes #4 / #11. Must precede any GPU spend; the labels are the ground truth.*
*Implemented as a sibling `verify_physics_labels.py` (four cases, production
camera), driving the **production** `simulate_clip` loop — not a copy — so the
verified path is the shipped path; `verify_collect_clip.py` stays a smoke test.
All checks are warmup-independent (an early draft tripped on a 1-render-step
physics warmup). Run results & what they caught:*

- *`offset_com` ✅ — `read_local_com` == authored offset to ~2 nm (was off by
  0.038 m). Found a real bug: **`MassAPI.centerOfMass` is authored in unscaled
  local space**, so a cuboid (scale=dims) shrank the offset ~0.4×. Fixed:
  `attach_center_of_mass` divides by `prim_local_scale`, and the label is now
  written from the post-reset measured `read_local_com` (`com_local_measured`),
  exact by construction. Also **resolved the `get_world_pose` origin-vs-CoM risk:
  it returns the prim origin → the `log_state_frame` reconstruction is correct,
  no double-count** (CoM world-xy is dead constant under spin).*
- *`two_body` ✅ — body-index ↔ label alignment holds.*
- *`spin_cuboid` ✅ — ang_vel logging perfect (pure +Z, exact direction) and
  quat composition correct. Before the fix it caught **PhysX's default ~0.05
  angular damping** (ω decayed ~1.7%/clip; linear damping was 0). Fixed:
  `zero_damping` on every dynamic body — controlled physics (gravity + initial v
  + contact only), conserves angular momentum through the `collide` mass signal.
  Re-run: ω pinned at 4.0, `angvel_camera_max_err` 0.0726 → 2.4e-5, **pass**.*

*Inertia is no longer authored (PhysX computes it; scale-correct). The
Monte-Carlo-validated `inertia_diagonal` is retained for reference only.*

Extend `verify_collect_clip.py` (or a sibling) beyond the single CoM=0 sphere to
the cases that exercise the risky label code, each with closed-form truth:

- **Spinning uniform cuboid, known constant ω** about an edge axis → validates
  `R_camera_world @ ang_vel_world` sign/scale and `quat_wxyz_mul` composition.
- **Authored-offset-CoM body** (from W2) → validates `R_body_world @ com_local`
  against `pos + R·offset`, and **resolves the `get_world_pose` origin-vs-CoM
  question** by checking the logged `com_world` against the analytic rigid-body
  trajectory under the authored (mass, CoM, inertia).
- **Two-body scene** → validates body-index ↔ label alignment (no index swap).
- **Production camera pose** `(2.2,2.2,1.8)` look-at `(0,0,0.1)` → so verify tests
  what we actually ship, not a bespoke broadside camera (#11).

Out of scope (no clean GT): general tumbling, PhysX CoM on real meshes.

### W5 — Composite glued bodies ✅ DONE — label gate + R8 colour check passed
*Code in `body_factory.spawn_composite_body` + `_partition_cuboid` +
`spawn_dynamic_body` dispatch; wired into `build_random_bodies` and into
`topple`/`collide`; verify case `composite` added to `verify_physics_labels.py`.*

**Label gate (W5 verify) PASSED:** PhysX integrates the per-slab densities
exactly as the analytic `ΣρV` label assumes — `mass_rel_err = 8e-9`,
`com_label_err = 5e-9 m` (5 nm), on a 2-slab box with a 42 mm CoM offset. So
per-shape-density composition into one rigid body is confirmed (no body-prim-mass
fallback needed). A `push` smoke run reproduced this on the production path
(com/mass err 0.0 across 14 composites, composite share 33% ≈ `composite_frac`).

**Material-binding fix (from the smoke run):** OmniPBR bound on the parent Xform
did NOT propagate to the child Cubes for RTX rendering — composites rendered
default grey (seg masks: blue body rendered ≈grey vs primitives keeping their
hue). That grey is itself an R8 cue ("grey box == composite"), so the visual +
physics materials are now bound to **each child gprim** (one shared material →
still no per-slab cue, but the colour renders; physics material on the collider
is also where PhysX reads it).

**Re-run confirms the fix:** composites now render their sampled colour across
scenarios (settle teal, drop green, topple teal/purple). R8 check — composite vs
primitive *cuboid* rendered saturation is indistinguishable in the free-object
set (0.40 vs 0.38, n=40/31); pooled+brightness-controlled there's a small ~0.04
gap (composite 0.32 vs primitive 0.36) — accepted as negligible against the
lighting/colour DR variance (likely a geometric shadowed-face effect from leaning
slabs, not a material gap; a `displayColor` fallback on the child Cubes is the
escape hatch if it ever proves to be a learnable cue). `collide` composite path
not separately re-run (low risk — same `spawn_dynamic_body` dispatch as the
confirmed `topple`/free paths). **W5 is the last item of the W1–W5
physics-fidelity sub-plan; the sub-plan is complete.** Mesh-asset bodies remain
as the deferred **W5b**.

A composite is a **uniform-colour cuboid internally split into 2–3 contiguous
density slabs** along one random axis (`config.composite_frac=0.3`,
`composite_n_parts_range`, `composite_split_min_frac`). Each slab is a child
`UsdGeom.Cube` collider carrying its own `density` (reuses `density_range`);
PhysX integrates the union into the body mass / CoM / inertia at `world.reset()`.
The parent Xform is the single rigid body (`RigidBodyAPI` + `RigidPrim`,
mirroring `load_dynamic_asset`); one OmniPBR + one PhysicsMaterial are bound on
the parent and inherited by the slabs, so the **density asymmetry is invisible**
(R8 — CoM recoverable from geometry + dynamics only, never pixels). Decision
change vs the original sketch below: parts are **uniform-coloured, not visually
correlated with CoM** (the user's call — no appearance cue).

- **Mechanism — slabs that tile a cuboid bbox**, chosen so the union *is* a
  cuboid: a composite advertises `shape="cuboid"`, `dims=bbox`, so every existing
  placement / lean / `preroll` path treats it exactly like a primitive cuboid and
  **no call-site geometry math changed** — only the spawn dispatch.
- **GT is closed-form for mass + CoM** (stronger than the "consistency only" this
  section first assumed): non-overlapping slabs give `mass = Σ ρ_k V_k`,
  `com_local = Σ ρ_k V_k t_k / mass` (logged as `com_local_analytic`; per-slab
  `parts` logged too). The `composite` verify case asserts PhysX's post-reset
  `get_mass` / `read_local_com` equal that analytic form — proving the one
  uncertain behaviour (per-shape-density composition into one rigid body).
  **Inertia stays PhysX-computed and unverified** (no clean closed form).
- **Everywhere incl. topple/collide** (user's call): in `topple` the invisible
  split biases *which face* the body lands on; in `collide` differing composite
  masses make the mass ratio observable.
- **Mesh-asset bodies deferred to W5b.** `asset_loader.load_dynamic_asset`
  exists, but a mesh's CoM/inertia is PhysX-computed with no closed-form to
  verify — a weaker GT tier. Ship the provable authored-composite core first.
- Determinism: the per-body composite draw shifts the RNG stream, so old seeds
  re-render (acceptable — W5 changes dynamics by design, regenerate-and-rerun).

## Correctness risk — RESOLVED (W4 offset_com)

`log_state_frame` reconstructs `com_world = get_world_pose() + R_body_world @
com_local`. The open question was whether `get_world_pose()` returns the prim
origin or the CoM (the latter would **double-count** the offset). **Settled: it
returns the prim origin** — `RigidPrimView.get_world_poses()` reads the PhysX
actor transform (`isaacsim.core.prims/.../rigid_prim.py`), and empirically the
reconstructed CoM world-xy is dead constant under spin (no wobble). The
reconstruction is correct. (Two *other* bugs surfaced instead — the unscaled
`centerOfMass` and default angular damping — both fixed; see W4.)

## Ground-truth contract

| Quantity | Phase 1 (authored) | Phase 2 (composite, W5) |
|---|---|---|
| density / mass | authored, exact | per-slab density authored → **exact** composite mass `ΣρV` (closed-form verify) |
| CoM offset | authored, exact (closed-form verify) | **exact** `ΣρV·t/ΣρV` from non-overlapping slabs (closed-form verify vs `read_local_com`) |
| inertia | PhysX-computed (scale-correct) | PhysX-computed, **unverified** (no clean closed form) |
| ω, v, trajectory | sim ground truth (verify round-trip) | sim ground truth |

## Sequencing & gating

- W1 + W2 + W3 **change dynamics → must precede P1 scale generation** (a batch made
  before them must be regenerated). They also gate the F-config / observability
  loss (P4) being meaningful.
- W4 **precedes any GPU spend** (GATE A feasibility runs on these labels).
- W5 ✅ done (was post-trust, parallel to early model work); W1–W5 complete.
- Resolution (§14) and retry (#3) from the critique are independent and can land
  alongside, but are not blockers for *this* fidelity pass.

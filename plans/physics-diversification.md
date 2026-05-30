# Physics Diversification & Verify Hardening

*Date: 2026-05-30. The data-gen fidelity pass that must land **before** any scale
generation (P1) and before the feasibility/architecture gates lean on physics
labels. Companion to [data_gen_critique0_analysis.md](data_gen_critique0_analysis.md)
(#4, #7) and [observability-weighted-loss.md](observability-weighted-loss.md)
(R6). Tracked operationally in [IMPLEMENTATION.md](IMPLEMENTATION.md) P0 hardening.*

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

### W3 — Observability scenarios ✅ DONE
*Fixes the R6 no-signal-in-data half. New builders in `scenarios.py` + registry.*
*Implemented: `build_topple` + `build_collide`, registered in `BUILDERS` and
`ClipConfig.scenarios`; per-scenario `observability_hint` tag injected in
`scenarios.build`. Per-body construction refactored into `body_factory.spawn_body`
(shared with `build_random_bodies`, RNG order preserved). `topple`'s CoM-face
signal is dormant until W2. Compile-checked only — not yet run under Isaac.*

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
- *`spin_cuboid` ⚠️ — ang_vel logging perfect (pure +Z, exact direction) and
  quat composition correct, but caught **PhysX's default ~0.05 angular damping**
  (ω decayed ~1.7%/clip; linear damping was 0). Fixed: `zero_damping` on every
  dynamic body — controlled physics (gravity + initial v + contact only),
  conserves angular momentum through the `collide` mass signal. Re-run: ω pinned
  at 4.0, `angvel_camera_max_err` 0.0726 → 2.4e-5, pass.*

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

### W5 — Composite glued bodies *(Phase 2, after W1–W4 trusted)*

- Fuse 2–3 primitives of differing density into one rigid body → emergent offset
  CoM that PhysX computes and `read_local_com` reads. Appearance partially
  correlates with CoM (more realistic than authored). GT = PhysX-computed (verify
  the measured-vs-physical consistency, not a closed form). Parked until the
  offset-CoM path is proven on the exact-GT authored case.

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

| Quantity | Phase 1 (authored) | Phase 2 (composite) |
|---|---|---|
| density / mass | authored, exact | per-part authored → exact composite mass |
| CoM offset | authored, exact (closed-form verify) | PhysX-computed (`read_local_com`, consistency check) |
| inertia | authored, exact | PhysX-computed |
| ω, v, trajectory | sim ground truth (verify round-trip) | sim ground truth |

## Sequencing & gating

- W1 + W2 + W3 **change dynamics → must precede P1 scale generation** (a batch made
  before them must be regenerated). They also gate the F-config / observability
  loss (P4) being meaningful.
- W4 **precedes any GPU spend** (GATE A feasibility runs on these labels).
- W5 is post-trust, parallel to early model work.
- Resolution (§14) and retry (#3) from the critique are independent and can land
  alongside, but are not blockers for *this* fidelity pass.

# W5 — Composite glued bodies (physics-diversification.md)

> **STATUS: ✅ DONE** (implementation plan below; kept as the record of intent).
> Shipped in `body_factory.spawn_composite_body`/`_partition_cuboid`/
> `spawn_dynamic_body`, wired into `build_random_bodies` + `topple`/`collide`,
> with a `composite` verify case. Results live in
> [physics-diversification.md §W5](physics-diversification.md). **As-built
> deviation from §2/§5 below:** the visual+physics materials are bound to **each
> child Cube**, not the parent Xform — parent-Xform binding does not propagate to
> the child gprims for RTX rendering (a smoke run showed grey composites), so the
> §Risks "bind per child" fallback became the actual approach. Label gate passed
> exactly (mass err 8e-9, CoM err 5 nm); R8 colour check passed (composite vs
> primitive-cuboid saturation 0.40 vs 0.38).

## Context

Every dynamic body today is **one primitive** with either a uniform CoM or a
single authored offset (W2, `body_factory.spawn_body`). W5 is the last item of
the physics-fidelity sub-plan: build rigid bodies out of **2–3 fused primitives
of differing density** so the offset CoM *emerges* from a real mass distribution
that PhysX integrates, rather than being authored onto one shape. This makes the
CoM/mass labels more physically realistic and gives the dataset compound-inertia
variety the single-primitive path can't produce.

Two constraints fixed with the user:
- **No appearance cue (R8).** The whole composite is one uniform colour/material,
  so density asymmetry is invisible. The CoM offset is recoverable only from
  **geometry + dynamics** (tipping/tumbling/impact), never from pixels.
- **Composites appear in every scenario**, including `topple` and `collide`,
  mixed with the existing W2 authored-offset primitives (they are not replaced —
  W2 stays as the pure-control half of the corpus).

**Mesh assets are explicitly out of scope for W5** (deferred to a later "W5b").
`asset_loader.load_dynamic_asset` already exists, but a mesh's CoM/inertia is
PhysX-computed with **no closed-form to verify against** — a strictly weaker
ground-truth tier than authored composites. W5 ships the provable-label core
first.

## Key design decision: a composite is a uniform cuboid that is internally density-split

The cleanest mechanism that is exact, R8-clean, and a drop-in everywhere:

- A composite occupies a **cuboid bounding box** and is built by splitting that
  box into `n∈{2,3}` **contiguous, non-overlapping axis-aligned slabs** along one
  random axis. Each slab is a child `UsdGeom.Cube` collider with its **own
  density**; the union exactly fills the bbox.
- Externally it looks like an ordinary uniform cuboid (one colour). Internally
  the heavy slab pulls the CoM off-centre — pure dynamics signal, zero pixel cue.
- Because the union *is* a cuboid, the composite advertises `shape="cuboid"`,
  `dims=bbox` to all existing placement / lean / spacing logic
  (`bounding_radius`, `half_extents`, `preroll_trajectory`), so **no call site's
  geometry math changes** — only the spawn call differs.

Ground truth is closed-form (matches the doc's Phase-2 contract row):
`mass = Σ ρ_k V_k`, `com_local = Σ ρ_k V_k t_k / mass` (slab `k` centred at `t_k`,
`V_k = dx_k·dy_k·dz_k`). Inertia stays PhysX-computed and **unverified** (no clean
closed form — same scoping as general tumbling).

*(Slabs-in-a-box keeps centroid at the box centre, so 100% of the CoM offset is
density/dynamics. Asymmetric compound silhouettes — where geometry itself cues the
centroid — are a possible later extension; noted, not in this cut.)*

## Files to change

### 1. `src/data_gen/config.py` — `ClipConfig` fields
Add, near the W1/W2 density/CoM fields:
- `composite_frac: float = 0.3` — per-body probability of being a composite.
- `composite_n_parts_range: tuple = (2, 3)` — inclusive slab count.
- `composite_split_min_frac: float = 0.2` — min fraction of the split axis any
  single slab gets (avoids degenerate slivers).
- Per-slab density reuses the existing `density_range` (200–8000) — the wide
  range is what creates a real, invisible mass asymmetry.

### 2. `src/data_gen/body_factory.py` — the construction path (the meat)
New `spawn_composite_body(world, rng, config, i, dims, position, color=None, orientation=None) -> (body, spec)`:
- Build a parent `UsdGeom.Xform` at `/World/body_{i:02d}`; partition the cuboid
  `dims` into `n` contiguous slabs along a random axis (split fractions sampled,
  clamped by `composite_split_min_frac`). For each slab `k`:
  - `UsdGeom.Cube` child at `/World/body_{i:02d}/part_{k}`, `size=1`,
    `xformOp:scale = slab_dims`, `xformOp:translate = t_k`.
  - `UsdPhysics.CollisionAPI.Apply` + reuse **`attach_density(child, ρ_k)`** with
    a per-slab `ρ_k ~ U(density_range)`.
- Wrap the parent with `RigidPrim(prim_path, name, position, orientation)` and
  `world.scene.add(...)` → applies `RigidBodyAPI`, exposes the
  `get_world_pose/get_com/get_mass/get_*_velocity/set_*` interface
  `log_state_frame` needs.
- Reuse **`zero_damping(parent)`**, **`attach_physics_material(...)`** (friction/
  restitution), and one shared **`attach_pbr_material`/OmniPBR bound on the
  parent** so all slabs inherit a single colour (uniform appearance). One
  **`apply_class_label(parent, name)`** → the composite is one seg instance.
- Spec mirrors `spawn_body` plus: `kind="composite"`, `shape="cuboid"`,
  `dims=bbox`, `parts=[{shape,dims,offset_local,density,volume,mass}…]`,
  `mass` (= Σ analytic), `com_local_analytic` (closed form). Primitive specs get
  `kind="primitive"`. `com_local_measured`/`mass_measured` are still filled
  post-reset by the existing loop in `collect_clip` (the closed-form cross-check).

Add a small dispatch helper so the three call sites don't duplicate the branch:
`spawn_dynamic_body(..., composite: bool)` → calls `spawn_composite_body` or
`spawn_body`.

Update `build_random_bodies`: per body draw a composite flag (`rng.uniform() <
config.composite_frac`); **when composite, force `shape="cuboid"`** and sample a
cuboid `dims` so the pre-spawn `bounding_radius`/placement treat it as a cuboid;
then dispatch. (RNG-order/determinism note below.)

### 3. `src/data_gen/scenarios.py` — `topple` and `collide`
Both already call `spawn_body` directly in a per-body loop:
- `build_topple`: per body, draw the composite flag; if composite force
  `shape="cuboid"` (still tall via `_sample_tall_dims`), keep the existing
  `_topple_lean_pose` (it only needs `half_extents`, unchanged for a cuboid) and
  dispatch to `spawn_composite_body`. The invisible density split now biases
  *which face it lands on* — the W3+W5 payoff.
- `build_collide`: per body, draw the composite flag; if composite use a cuboid
  `dims`; dispatch. Differing composite masses make the **mass ratio** observable
  on impact.
- `cluttered`/`drop`/`roll`/… inherit composites automatically via
  `build_random_bodies` — no per-builder change.

### 4. `src/data_gen/verify_physics_labels.py` — W5 trust gate
Add a `composite` case in the style of the existing cases (production camera,
driving the real `simulate_clip`): a known 2-slab box (e.g. 2:1 split, ρ=8000 vs
ρ=1000) → assert `get_mass() ≈ ΣρV` and `read_local_com() ≈` the analytic
`ΣρV·t/ΣρV` (offset toward the steel slab), within tolerance. This is the gate
that proves PhysX integrates per-slab density into the composite mass/CoM the way
the analytic label assumes — the one genuinely uncertain runtime behaviour.

### 5. Docs (source of truth — update in the same change)
- `plans/physics-diversification.md` W5: mark in-progress → done after the Isaac
  run; record the slabs-in-a-box mechanism and the verify result.
- `plans/IMPLEMENTATION.md` P0: add a W5 line; note assets deferred to W5b.

## Risks / things only an Isaac run can confirm
- **Per-shape density composition** — that PhysX builds composite mass/CoM/inertia
  from child-collider `MassAPI` densities with no body-level mass authored. The
  verify case exists precisely to confirm this; if PhysX needs density on the body
  prim instead, the fix is local to `spawn_composite_body`.
- **Material inheritance** — OmniPBR bound on the parent Xform inheriting to child
  Cubes (uniform colour). Fallback: bind the same OmniPBR per child.
- **`RigidPrim` over a hand-built compound** — that `apply_physics_material` /
  velocity setters behave on a multi-collider body. (`load_dynamic_asset` already
  wraps multi-prim assets in `RigidPrim`, so this is precedented.)
- **Determinism**: inserting the per-body composite draw shifts the RNG stream, so
  existing seeds render different clips. Acceptable — no scale batch exists yet and
  W5 changes dynamics by design (regenerate-and-rerun, per the doc's sequencing).

## Verification
1. Compile-check: `python -c "import ast; [ast.parse(open(f).read()) for f in (...)]"`
   on the four edited modules (no Isaac import needed).
2. **W5 label gate** (must pass before trusting the data):
   `${ISAAC_SIM}/python.sh src/data_gen/verify_physics_labels.py` → the new
   `composite` case passes alongside the existing 3.
3. **End-to-end smoke** across scenarios incl. composites:
   `${ISAAC_SIM}/python.sh src/data_gen/collect_clip.py --result out/w5 --n_clips 8`
   then inspect: composite bodies appear (`meta.json` `bodies[].kind=="composite"`,
   non-trivial `com_local_analytic`), `com_local_measured ≈ com_local_analytic`,
   `mass_measured ≈ mass`, and per-scenario `--scenario topple` / `--scenario
   collide` produce visible motion. Confirm composites render as **uniform-colour
   boxes** (no slab seams / per-part colour) in `rgb.mp4`.

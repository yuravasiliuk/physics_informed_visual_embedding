# Scene Diversification — RigidProbe v0.1

**Extends:** [probe-dataset.md](probe-dataset.md) §7 (generation params) and §10 (versioning).
**Motivates:** the first generated clip from `collect_clip.py` showed a single scenario (objects falling vertically on an empty plane) under a single distant light — visually flat and motion-limited. The probe-task signal is monotonous: every clip's `lin_vel_camera` is dominated by gravity, every clip's background is identical.

This plan introduces **v0.1** — a richer scene generator that still uses the v0 schema, no field renames, but draws from six diversification axes including Isaac Sim's default asset library.

---

## 1. Diagnosis: what's monotonous today

| Symptom | Why it matters for the probe |
|---|---|
| Single archetype (drop) | `lin_vel_camera` is essentially `g·t·R_cw·ẑ_world + ε`. The encoder can predict it from the frame index alone — no actual physics decoding needed. |
| No static geometry | Bodies never interact with the scene — no rolling, no bouncing, no settling. ω is purely initial + air resistance. |
| Default ground plane + single distant light | Identical background across clips. The encoder cannot distinguish "lighting" from "object" — no scene cues for object segmentation beyond color. |
| Flat single-color ground | No texture on the floor → no parallax / translation anchor. A body sliding across a uniform plane gives the encoder almost no positional cue from the background; depth and normals of the ground are featureless. |
| Single hardcoded static camera `(2.2,2.2,1.8)→(0,0,0.1)` | Every clip shares one viewpoint and one projection. A frozen-encoder probe can latch onto the fixed projection; zero viewpoint generalization, and the sim→real story has no camera-pose variation. |
| Primitive shapes only | Visual distribution is far from anything photorealistic. Even if the probe works on this, the result tells us nothing about V-JEPA 2's actual physics encoding on realistic video. |
| Diffuse-only materials | No specular highlights, no reflections — no shading cues for surface orientation or material identity. |

The first three symptoms hurt physics signal directly. The last two hurt sim→real transfer of any probe finding.

---

## 2. Six orthogonal diversification axes

| Axis | Today | v0.1 target | Probe value |
|---|---|---|---|
| **A. Static scaffolding** | Ground plane only | Ramps, low walls, platforms, bowls (from primitives) | High — unlocks roll/slide/settle motion modes |
| **B. Initial motion regime** | Drop from rest, random ω | Throw (init lin vel), slide, placed-on-surface | High — kills the "everything is freefall" shortcut |
| **C. Lighting** | One default distant light | HDRI environment + random rotation, optional dome/sky light | Medium — affects appearance more than dynamics |
| **D. Ground / background** | Default plane (uniform) | Random PBR ground material (asphalt, wood, concrete, fabric) | Medium — gives texture cues for segmentation |
| **E. Material** | Diffuse + random HSV | PBR (random roughness, occasional metallic) | Low for v0.1; matters more at v1 |
| **F. Asset richness** *(new)* | Primitives only | Isaac default assets — pallets, crates, robotic arms (static), tool boxes — alongside primitives | **High** — closes the sim→real visual gap that the entire project hinges on |
| **G. Camera viewpoint** *(new, v0.2)* | One hardcoded static pose | Per-rollout rig of N jittered viewpoints on a shell around the scene; each view saved as an independent clip | **High** — kills the fixed-projection shortcut and yields free multi-view-consistent samples (see §10) |

Axes are independent; pick a subset per tier.

---

## 3. Scenario archetypes

Instead of a single scene template, each clip samples an archetype from a weighted distribution. Each archetype is a small builder function in `src/data_gen/scenarios.py` that returns `(bodies, body_specs, scene_specs)`.

| Archetype | Static geom | Initial state | Dominant motion | Asset use |
|---|---|---|---|---|
| `drop` (current) | none | bodies airborne, v=0 | freefall + spin | primitives |
| `drop_on_platform` | low platform (cuboid kinematic) | bodies airborne above platform | impact + bounce | primitive or asset platform |
| `roll` | ramp angled 15°–35° | bodies at rest on ramp | rolling + sliding | primitives on prim ramp |
| `throw` | none, optional far wall | bodies airborne with init v ≠ 0 | parabolic + spin | primitives |
| `settle` | bowl / open box | bodies airborne above container | settling + pile-up | primitive bowl or asset crate |
| `push` | flat ground | bodies at rest, init v along ground | sliding + rolling friction | primitives or small asset props |
| `cluttered` *(new with axis F)* | scene populated with Isaac static props (pallets, crates, shelves, robot arms) | bodies airborne above, falling between/onto props | bouncing + occlusion | **Isaac assets as static scaffolding + primitive or asset dynamic bodies** |

The `cluttered` archetype is the photorealism workhorse. It uses Isaac's existing furnished props instead of hand-built ramps and walls, giving each clip a different scene composition without us authoring USD by hand.

---

## 4. Isaac Sim asset library (axis F)

### What we use

Isaac Sim ships USD assets reachable through `omni.isaac.nucleus.get_assets_root_path()` (typically `omniverse://localhost/NVIDIA/Assets/Isaac/<version>/`). Three categories matter here:

| Category | Purpose | Example paths |
|---|---|---|
| **Static props** | Scene scaffolding for `cluttered` and `settle` archetypes | `Isaac/Props/KLT_Bin/small_KLT.usd`, `Isaac/Props/Pallet/pallet.usd`, `Isaac/Props/Rack/rack.usd`, `Isaac/Props/Mounts/ur10_mount.usd` |
| **Static robots** | High-visual-interest distractors / occluders, fixed pose | `Isaac/Robots/Franka/franka.usd`, `Isaac/Robots/UR10/ur10.usd`, `Isaac/Robots/AnymalC/anymal_c.usd` (all base-fixed) |
| **Dynamic props** | Photorealistic small bodies, drop-in replacement for primitives | `Isaac/Props/YCB/Axis_Aligned/*.usd` (the YCB subset Isaac ships), `Isaac/Props/Blocks/*.usd`, `Isaac/Props/Apartment/*.usd` (subset) |

The exact paths depend on the installed Isaac version — the asset library is a runtime-discovered list, not a hard-coded enum.

### How we load them

```python
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

assets_root = get_assets_root_path()
asset_url = f"{assets_root}/Isaac/Props/Pallet/pallet.usd"
add_reference_to_stage(usd_path=asset_url, prim_path="/World/pallet_01")
```

After loading, apply physics + collision APIs if not already authored on the asset:

```python
from pxr import UsdPhysics, PhysxSchema
prim = stage.GetPrimAtPath("/World/dynamic_box_01")
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.MassAPI.Apply(prim)
UsdPhysics.CollisionAPI.Apply(prim)
# For convex-hull collision on non-convex meshes:
PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
```

Static props get only `CollisionAPI` (no `RigidBodyAPI` → no dynamics, just a collider).

### Three gotchas that must be handled

1. **CoM ≠ prim origin for asymmetric assets.** The current `log_state_frame` writes `com_world = prim_origin`, justified by "uniform-density primitives." This is wrong for a Franka arm, a pallet, or a YCB mug. Fix: read CoM via the rigid body view (`rigid_prim_view.get_world_poses()` + `get_coms()`), or query `UsdPhysics.MassAPI.GetCenterOfMassAttr()` directly. The packed-state refactor (probe-dataset.md §B7 in the audit) is a good time to bake this in.

2. **Mass and inertia defaults can be wrong.** Many shipped USD assets have placeholder mass (e.g., 1 kg) or no `MassAPI` at all. Either:
   - Trust the asset (acceptable for the probe — we don't need physically accurate dynamics, we need *visually consistent* dynamics).
   - Override mass via `MassAPI.GetMassAttr().Set(...)` with a value sampled from a plausible range per category.
   - Recommendation: trust for v0.1, override for v1 when we care about transfer fidelity.

3. **Segmentation IDs span more prims.** A Franka arm is ~12 articulated links — instance segmentation may give each link a different ID. For the probe, we want one ID per *body of interest*. Fix: at load time, walk the asset hierarchy and tag the root rigid body with a `semantic_label`; configure the seg annotator to group by semantic label instead of per-prim instance. (This is how Isaac's `replicator.semantics` API is designed to work.)

### What we explicitly do NOT do in v0.1

- **No articulated dynamics.** Robots are loaded with all joints fixed — they're occluders, not actors. Articulation adds a second physics state variable (joint angles) that's outside the probe's scope.
- **No conveyors, no actuated assets.** Same reason.
- **No external asset packs** (YCB, GSO downloads). v0.1 uses only what Isaac ships. External packs are a v1 concern with license/storage trade-offs.

---

## 5. Tiered rollout

### Tier 1 — Cheap diversification (½–1 day)

Pure primitives + Isaac default lighting infrastructure. No external assets yet.

- Implement static-prim factories: `_make_ramp`, `_make_wall`, `_make_platform`, `_make_bowl` — all kinematic (no dynamics, just collision).
- Implement three scenarios: `drop`, `roll`, `throw`. Cover the three motion modes most underrepresented today.
- HDRI lighting via `rep.create.from_env_light(...)`. Ship 5–10 curated HDRIs from Polyhaven inside the repo (~20 MB total).
- `ClipConfig.scenarios: tuple = ("drop", "roll", "throw")` + `scenario_weights`.
- `meta.json` gains `scenario: str` and `hdri: {path, rotation_deg}`.

**Outcome:** physics signal stops being free-fall-dominated. Lighting starts varying.

### Tier 2 — Isaac asset integration (1–2 days)

This is the bulk of the photorealism gain.

- Asset registry module `src/data_gen/asset_library.py` — curated dict of `category → [usd paths]` discovered relative to `get_assets_root_path()`. Validates paths exist on import.
- Loader helpers: `_load_static_asset(stage, usd_url, prim_path, semantic_label)`, `_load_dynamic_asset(...)` with physics APIs applied.
- Fix CoM derivation in `log_state_frame` — read CoM from the rigid body view, not from prim origin. (Touches the packed-state refactor too.)
- Fix segmentation grouping — semantic-label-based instead of per-prim.
- New scenario `cluttered`: spawn 2–5 static Isaac assets (pallets, crates, shelves, robots) as scaffolding, then drop 3–8 dynamic bodies (mix of primitives + small Isaac props) above them.
- Add scenarios `drop_on_platform`, `settle`, `push`.
- `meta.json` per-body specs gain `asset_url: str | null` (null = primitive).

**Outcome:** clips look like real industrial / lab scenes. Probe results become meaningful for sim→real questions.

### Tier 3 — Material and ground (1 day)

- PBR materials via `OmniSurface`: random roughness ∈ [0.2, 0.9], occasional metallic ∈ {0.8, 1.0}.
- Random ground material from a curated set of 4–6 (asphalt, wood, concrete, fabric, tile).
- Optional "room" enclosures (back wall + side walls) at sampled distances.

**Outcome:** material/appearance variation. Stronger sim→real visual bridge.

### Tier 4 — Deferred to v1

- External asset packs (YCB / GSO downloads with license handling).
- Articulated robots in motion.
- Light color and intensity randomization beyond HDRI.
- Multi-camera per scene.
- Conveyors / actuated scene props.

---

## 6. The big call: regenerate v0, or call this v0.1?

The plan §10 currently scopes v0 to "Primitives, fixed camera, fixed lighting." This diversification takes us strictly past that line. Two options:

**Option A — Promote Tier 1 + Tier 2 into v0.** Don't ship the current monotonous corpus to the V-JEPA 2 probe at all; generate v0 against the diversified spec from the start.
- **Pro:** the probe number we get measures something real and transferable.
- **Pro:** no wasted compute on a corpus we'll throw away.
- **Con:** delays the probe by 2–3 days of engineering.

**Option B — Keep current `collect_clip.py` as v0, label this v0.1.** Generate the monotonous corpus, run the probe, then redo on v0.1.
- **Pro:** faster to the first probe number.
- **Con:** if the probe passes on v0, we don't know whether it'd pass on a less-trivial distribution — almost certainly a misleading result.
- **Con:** wastes the corpus generation cost (~hours of Isaac time we have to redo).

**Recommendation: Option A.** The probe's value is information about what V-JEPA 2 actually encodes; a probe on free-fall-only data tells us almost nothing transferable. Spending 2–3 days to make the probe substrate meaningful is a much better ROI than racing to a misleading first number.

This means: do not run `collect_shard.py` (step 7 in probe-dataset.md §11) until Tier 1 + Tier 2 land.

---

## 7. Implementation order

Adds onto [probe-dataset.md §11](probe-dataset.md#11-implementation-order). Insert between current step 5 (sanity checks) and step 6 (WebDataset writer).

1. **5.1 Static prim factories** (`_make_ramp`, `_make_wall`, `_make_platform`, `_make_bowl`) in a new `src/data_gen/static_geometry.py`. Pure Isaac primitives, kinematic, with `UsdPhysics.CollisionAPI` only.

2. **5.2 Scenario module.** Create `src/data_gen/scenarios.py`. Each archetype is `build_<name>(world, rng, config) -> (bodies, body_specs, scene_specs)`. `scene_specs` is a small dict capturing the static geometry choices (ramp angle, platform position, etc.) for inclusion in `meta.json`.

3. **5.3 Dispatch from `build_scene`.** `build_scene` becomes thin: sample archetype, call the right builder, return.

4. **5.4 HDRI lighting.** New helper `src/data_gen/lighting.py`. Loads one HDRI from a configured set, random rotation. Polyhaven HDRIs vendored under `assets/hdri/`.

5. **5.5 Asset library.** `src/data_gen/asset_library.py` — discovers Isaac asset paths via `get_assets_root_path()`, organizes by category. On import, validates paths resolve (warn if some don't, drop them from the active set rather than crashing).

6. **5.6 Asset loaders.** `_load_static_asset`, `_load_dynamic_asset` — apply physics APIs as needed, attach semantic labels for segmentation.

7. **5.7 CoM fix.** Replace `prim_origin == CoM` assumption in `log_state_frame`. Use `RigidPrimView` or `MassAPI` to read true CoM. This is the "step B7 + asset support" combined.

8. **5.8 Segmentation grouping.** Configure the seg annotator (or post-process the seg output) so all prims under one semantic root collapse to one ID. Unblocks the deferred §9 sanity check.

9. **5.9 `cluttered` scenario.** Spawn 2–5 static Isaac assets + 3–8 dynamic bodies above. Pose rejection sampling extended to consider the static geometry's footprint.

10. **5.10 Tier 3 materials + ground** (post-cluttered, before shard generation). Lower priority.

After 5.1–5.9 land, resume probe-dataset.md §11 step 6 (WebDataset writer) with confidence that the substrate is meaningful.

---

## 8. Schema additions for v0.1

Backwards-compatible additions to `meta.json` only — no field renames:

```jsonc
{
  // ... existing v0 fields
  "scenario": "cluttered",
  "scene": {
    "hdri": {"path": "assets/hdri/studio_small_03.hdr", "rotation_deg": 137.4},
    "static_assets": [
      {"prim_path": "/World/pallet_01", "asset_url": ".../Isaac/Props/Pallet/pallet.usd"},
      {"prim_path": "/World/rack_01",   "asset_url": ".../Isaac/Props/Rack/rack.usd"}
    ],
    "scenario_specs": {"ramp_angle_deg": 25.0}   // archetype-specific, free-form
  },
  // v0.2 multi-view provenance (axis G, §10). Each clip is still single-camera:
  // its meta already carries camera_position_world / camera_look_at_world /
  // R_camera_world for THIS view. `multiview` only links the N sibling clips
  // that share one physics rollout — independent on disk, groupable by metadata.
  "multiview": {
    "rollout_seed": 1234,    // identical across the N siblings of one rollout
    "camera_index": 3,       // 0..n_cameras-1
    "n_cameras": 7
  },
  "bodies": [
    {
      "id": 1, "name": "body_00", "shape": "asset",
      "asset_url": ".../Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
      "dims": null,                              // assets don't have synthetic dims
      "color_rgb": null,                         // asset material drives appearance
      "static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.2,
      "init_position_world": [0.1, 0.0, 0.8],
      "mass_kg": 0.451                            // read back from asset
    }
  ]
}
```

For primitive bodies, `asset_url` is null and `shape` stays one of `cuboid/sphere/cylinder/capsule`. For asset bodies, `shape: "asset"` is a sentinel and the URL identifies the geometry.

`schema_version` stays `"rigid-probe/v0-draft"`. When v0.1 corpus is generated, bump to `"rigid-probe/v0.1"`.

---

## 9. Open questions

1. **HDRI vendoring vs runtime download.** Vendoring 5–10 HDRIs adds ~20 MB to the repo (manageable). Alternative: download on first run from Polyhaven. Vendoring is simpler and offline-safe.
2. **Asset-discovery on import vs lazy.** Asset library validates paths on import → fails fast if Isaac assets root is unreachable. Lazy = silent failure until first use. Recommend eager.
3. **Mass override policy.** Trust shipped USD masses or override per category? Recommend trust for v0.1, revisit for v1.
4. **`cluttered` weighting.** What fraction of v0.1 should be `cluttered`? Tradeoff: more cluttered → more photorealistic but slower to generate and harder to validate. Start at 30%; tune by inspection.
5. **License audit for Isaac default assets.** NVIDIA's standard Omniverse license generally allows research use, but worth confirming before any public release.

---

## 10. Camera viewpoint diversity + multi-view capture (axis G, v0.2)

*Addresses the hardcoded-camera and flat-ground symptoms in §1. Decided with the
user 2026-05-30. Companion to the physics fidelity pass
([[physics-diversification]] W1–W4).*

### 10.1 Multi-camera fan-out — N independent clips per physics rollout

The key efficiency lever: the **expensive** work per clip is the fixed cost
(Isaac boot, scene build, asset load, physics warmup, the physics rollout
itself); **rendering a frozen scene from another pose is cheap**. So one physics
rollout is rendered from `n_cameras` viewpoints at once and **fanned out into
`n_cameras` independent clips**.

- **Physics steps once per frame.** Each frame: `world.step()` → one
  `rep.orchestrator.step()` flushes all N render products → for each camera,
  `capture()` its annotators and `log_state_frame()` with *that* camera's
  extrinsics, on the same physics state.
- **Each camera = a standalone clip in its own folder**, byte-for-byte the same
  layout as a single-camera clip (own `rgb.mp4` / `depth.npz` / `normals.npz` /
  `seg.npz` / `state.json` / `meta.json`). **Decision (user): the N siblings are
  NOT bundled into one folder.** A rollout of N cameras over M rollouts yields
  `M·N` top-level clip folders.
- **The per-clip contract is therefore unchanged.** `state.json` stays in *that
  camera's* frame; `meta.json` carries that camera's single `R_camera_world`.
  No world-frame migration; the W4 verify (single-camera reconstruction) is
  untouched. The only addition is the `multiview` provenance block (§8) so the
  siblings can be regrouped later for multi-view / viewpoint-invariance work.
- **Why not just generate N independent clips?** Because these N share *identical
  dynamics* — exactly the "same physics, different viewpoint" signal the
  domain-marginalization regularizer (P3) and a 3D-aware frozen encoder want, and
  which independent rollouts cannot provide.

### 10.2 Camera rig — predefined viewpoints + per-rollout jitter

A canonical rig of `n_cameras` viewpoints on a shell around the scene centroid,
jittered per rollout (user's "predefined set, jitter with reasonable noise"):

- Spherical params per camera: **azimuth** spread evenly (≈`360/n` apart),
  **elevation** spread across a band (≈20–55°, always positive so the camera
  looks *down* onto the cluster and never sits under the floor), **radius**
  ≈2.8–3.6 m, **look-at** near the scene centroid `(0,0,~0.12)`.
- Per-rollout jitter: ±azimuth, ±elevation (clamped), ±radius, ±look-at, so no
  two rollouts share a projection while the rig stays well-conditioned.
- `world_to_camera_frame` gets a **degenerate-camera guard** (critique #12):
  reject/clamp any view whose forward is near-parallel to world-up (NaN in the
  `cross(forward, up)`), which the higher rig elevations could otherwise trip.

### 10.3 Visibility / occlusion (user requirement: "object must be visible")

Two layers, in order of cost:

1. **Geometric** — inward-looking shell at positive elevation means cameras look
   down onto the object cluster near the origin; they cannot be inside geometry
   or below ground. Bounded radius keeps the scene in FOV.
2. **Per-camera validation** — generalize the existing `validate_clip`
   foreground/`empty_seg` check to *per camera*: a view whose target-pixel
   coverage is too low (a body fully occluded behind a `cluttered` wall) fails
   validation for *its* clip only; the sibling views are unaffected.
   *v0.2 scope:* post-hoc per-camera validation (failed view → `validation.ok=False`).
   *Follow-up (noted, not in v0.2):* a t=0 pre-flight render that re-jitters a
   camera whose start coverage is poor, to guarantee N good views per rollout.

### 10.4 Ground texture — visible grid (axis D, concrete choice)

The flat-color ground gives no spatial anchor. **Decision (user): grid-line
floor** (the Isaac-default-grid aesthetic, but generated locally — the shipped
default ground hits the asset server, which our `/Isaac`-only mirror lacks).

- Generate a grid-line texture PNG locally (numpy + cv2), cache it, and bind it
  to the `GroundPlane` via `OmniPBR(texture_path=…, texture_scale=…)`
  (`GroundPlane(visual_material=…)` / `apply_visual_material` — confirmed API).
- Randomize per clip via `texture_scale` (apparent line spacing) + base-color
  tint, for diversity without regenerating the texture.
- Suppressed when an environment USD ships its own floor (as today).

### 10.5 Budget & cost

- VRAM/throughput scale with **render resolution × n_cameras** (render res, not
  `save_resolution`, is what sits in VRAM). Target box: **24 GB, headless**.
- **Start at `n_cameras = 7`** (user). Tune count / render res after the first
  VRAM check; engage the §14 `save_resolution` drop to keep disk in check
  (`M·N` clips multiplies storage).
- Transient host RAM holds all N cameras' frames for one rollout (physics
  produces them interleaved per timestep), then writes and frees per camera.

---

## Relation to existing plans

- **[[probe-dataset]]**: this plan extends §7 and §10, inserts steps 5.1–5.10 into §11. No field renames; only additions to `meta.json`.
- **[[synthetic-data-collection]]**: the broader sim+real data strategy. Diversification feeds in here as "what the sim half looks like."
- **[[training-data]]**: the encoder will train on this corpus. Richer scenes mean the encoder can't shortcut on a single drop manifold.
- **[[critique2]]**: directly addresses the critic's "Phase 0.5 is too easy" point. A diversified corpus makes the probe a meaningful test rather than a trivial one.

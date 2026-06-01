# Isaac Asset & Environment Diversification — RigidProbe v0.1

**Extends:** [scene-diversification.md](scene-diversification.md) §4 (Isaac
Sim asset library).
**Motivates:** Smoke-test inspection of the `cluttered` scenario shows only
one prop type appearing — pallets dominate every clip. Beyond that, every
clip renders against the same procedural grey floor + flat distant-light
backdrop. There's no sense of *where* the scene is happening (warehouse?
office? lab?). Both axes — prop variety and *environmental* context — are
under-utilised relative to what Isaac Sim ships.

This plan covers both: a wider prop pool **and** loadable scene backdrops
(`Simple_Warehouse`, `Simple_Room`, `Office`). Articulated robots are
explicitly out of scope for v0.1.

---

## 1. Diagnosis

| Symptom | Cause |
|---|---|
| Only pallets visible in `cluttered` | `CANDIDATE_STATIC_PROPS` has 5 paths; two (`/Rack/rack.usd`, `/Mounts/Stand/stand_instanceable.usd`) don't exist locally and silently drop. Of 3 remaining, pallet is largest and visually dominates KLT bin / ur10_mount |
| No YCB, blocks, mugs, etc. ever render | `library.sample_dynamic_prop` is unused; `cluttered` drops *primitives*, not assets. 50 YCB items + 24 blocks + 8 mugs sit idle |
| Every clip's backdrop is the same | No environment USD is loaded; we render a procedural `GroundPlane` against a single `DistantLight` |
| Asset coverage tiny | 12 wired / 3,444 available (0.3%); 0 environments wired / 11 available on S3 |

---

## 2. What's available

### Props (already in local mirror, 3,444 USDs)

| Subtree | Count | Probe relevance |
|---|---|---|
| `Props/YCB/Axis_Aligned` | 50 | **High** — canonical dynamic bodies for vision research |
| `Props/Blocks` | 24 | **High** — clean colored cubes |
| `Props/Mounts` | 23 | Medium — tables, stands, screws |
| `Props/Sektion_Cabinet` | 10 | Medium — kitchen cabinet (scenery) |
| `Props/Pallet` | 8 | High — currently used; needs siblings (pallet_holder, etc.) |
| `Props/Mugs` | 8 | High — irregular shape, asymmetric CoM |
| `Props/KLT_Bin` | 6 | High — useful as bowl / settle container |
| `Props/Forklift`, `Props/Dolly` | 4 each | Medium — wheeled scenery |
| `Props/Rubiks_Cube`, `Props/Food` | 2-4 | Medium — textured small bodies |
| `Props/Conveyors`, `Props/Factory` | 142, 84 | Low for v0.1 — articulated / complex; defer |

### Environments (on S3, NOT yet in local mirror)

| Subtree | Files | Variants | Relevance |
|---|---|---|---|
| `Environments/Simple_Warehouse` | 334 (~497 MB) | `warehouse.usd`, `full_warehouse.usd`, `warehouse_with_forklifts.usd`, `warehouse_multiple_shelves.usd` | **High** — direct "storage / warehouse" backdrop the prompt asks for |
| `Environments/Simple_Room` | 89 (~116 MB) | `simple_room.usd` | **High** — generic interior |
| `Environments/Office` | 21 (~0.1 MB; references materials) | `office.usd` | **High** — office backdrop |
| `Environments/Hospital` | ? | `hospital.usd` | Medium — clinical interior; adds variety |
| `Environments/Grid` | 3 (small) | `default_environment.usd`, `gridroom_curved.usd`, `gridroom_black.usd` | Low — replaces what we already do procedurally |
| `Environments/Outdoor`, `Terrains` | ? | terrains, outdoor | Out of scope for v0.1 |

Estimated extra sync: ~700 MB for the four high-relevance subtrees plus
the materials they reference under `/Isaac/Materials`.

---

## 3. Diversification axes (revised)

| Axis | Today | v0.1 target | Cost |
|---|---|---|---|
| **A.** Static prop variety | 3-4 paths | ~18 paths across containers / tables / cabinets / dollies | XS — list extension |
| **B.** Environment backdrops *(NEW, prompt-driven)* | none (procedural ground only) | sample one of: none / warehouse-3 variants / simple_room / office | **M** — sync + loader + per-env camera tweaks |
| **C.** Dynamic asset bodies | 0 (primitives only) | YCB subset + Blocks + Mugs as drop-in body sources | M — `body_factory` extension + spec schema |
| **D.** Per-scenario asset substitution | none | pallet as `drop_on_platform`; KLT bin as `settle` bowl | M |
| **E.** Per-instance scale jitter | none | log-uniform ±15% on asset spawn | XS |
| **F.** Per-category sampling weights | uniform | weighted (large items rarer) | XS |
| ~~Robotic occluders~~ | — | **dropped per direction — too much articulation complexity for v0.1** | — |

---

## 4. Proposed taxonomy

Replace the flat `(static_props, static_robots, dynamic_props)` triple in
`asset_library.py` with a category-keyed structure so scenarios can ask
for specific *kinds* of assets:

```python
@dataclass(frozen=True)
class AssetCategory:
    name: str
    paths: tuple        # validated URLs
    role: str           # "static_scenery" | "dynamic_body" | "environment"
    weight: float = 1.0
    typical_height: float = 0.3
```

Static scenery categories:

| Category | Subtree(s) | Examples |
|---|---|---|
| `boxes_containers` | Pallet, KLT_Bin, Sektion_Cabinet | pallet, pallet_holder, small_KLT, sektion_cabinet_instanceable |
| `industrial_scenery` | PackingTable, Dolly, Forklift | packing_table, dolly_main, forklift_main |
| `mounts_tables` | Mounts | table, textured_table, thor_table, ur10_mount |

Dynamic body categories:

| Category | Subtree | Examples |
|---|---|---|
| `ycb_food` | YCB/Axis_Aligned | cracker_box, sugar_box, mustard_bottle, banana, gelatin_box |
| `ycb_tools` | YCB/Axis_Aligned | power_drill, scissors, large_clamp, large_marker |
| `ycb_household` | YCB/Axis_Aligned | mug, bowl, foam_brick, wood_block, pitcher |
| `colored_blocks` | Blocks | red_block, blue_block, green_block, yellow_block, nvidia_cube |
| `kitchen_items` | Mugs, Food, Rubiks_Cube | SM_Mug_*, mac_n_cheese, rubiks_cube |

Environment categories (new):

| Category | Subtree | Variants | Notes |
|---|---|---|---|
| `warehouse` | Environments/Simple_Warehouse | warehouse, full_warehouse, warehouse_with_forklifts, warehouse_multiple_shelves | High shelving, industrial lighting; aisles ~2 m wide |
| `room` | Environments/Simple_Room | simple_room | Plain interior — neutral baseline |
| `office` | Environments/Office | office | Cubicles + desks — tight aisles |
| `none` | (procedural) | — | Existing behavior: procedural GroundPlane + fallback DistantLight |

---

## 5. Tiered rollout

### Tier 1 — Wider static-prop pool *(½ day; addresses "only pallet")*

1. Replace `CANDIDATE_STATIC_PROPS` with the union of `boxes_containers` + `mounts_tables` + `industrial_scenery` paths (~15-18 paths). Drop the non-resolving entries.
2. Add per-prop sampling weights (pallet=0.5, KLT=1.0, dolly=0.7, table=1.0, cabinet=0.8, …) so pallets no longer dominate.
3. Leave the flat-list internal structure for now — taxonomy refactor is Tier 3.

**Outcome:** `cluttered` clips visually vary across pallets, tables,
dollies, cabinets, KLT bins. No scenario code changes.

### Tier 2 — Environment backdrops *(1-2 days; the prompt's main ask)*

This is the highest-impact change — it gives every clip a sense of place.

1. **One-time asset sync** (manual step the user runs):
   ```bash
   aws s3 sync --no-sign-request \
     s3://omniverse-content-production/Assets/Isaac/5.1/Isaac/Environments/ \
     /home/yura/isaac_assets/Isaac/Environments/ \
     --exclude "Digital_Twin*" \
     --exclude "Jetracer*" \
     --exclude "Outdoor*" \
     --exclude "Terrains*" \
     --exclude "Hospital*"      # optional; ~150 MB clinical setting
   ```
   Then sync materials referenced by the envs:
   ```bash
   aws s3 sync --no-sign-request \
     s3://omniverse-content-production/Assets/Isaac/5.1/Isaac/Materials/ \
     /home/yura/isaac_assets/Isaac/Materials/
   ```
   Expected ~700 MB total. If `aws` is unavailable, an HTTP-based
   alternative using `curl` + the S3 REST endpoint can list and download
   each file individually.

2. **New module `src/data_gen/environments.py`** — env loader analogous to `lighting.py`:
   ```python
   ENV_URLS = {
       "warehouse":                "{root}/Isaac/Environments/Simple_Warehouse/warehouse.usd",
       "warehouse_full":           "{root}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
       "warehouse_with_forklifts": "{root}/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
       "warehouse_shelves":        "{root}/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
       "simple_room":              "{root}/Isaac/Environments/Simple_Room/simple_room.usd",
       "office":                   "{root}/Isaac/Environments/Office/office.usd",
   }
   def setup_random_environment(rng, config) -> dict | None: ...
   ```
   Each candidate is validated against the local mirror at import time
   (same pattern as `asset_library.discover`). Missing envs are dropped
   with a warning.

3. **Wire into `collect_clip.build_scene`**:
   - Sample `env_name = rng.choice(config.environments)`. If `"none"`, current behavior. Otherwise:
     - `add_reference_to_stage(env_url, "/World/environment")` — env brings its own floor and lights.
     - **Skip** the procedural `GroundPlane` (env has a floor already).
     - **Skip** the fallback `DistantLight` from `lighting.setup_random_hdri` (env's lighting is rich enough).
   - Record under `scene_specs["environment"] = {"name": ..., "url": ...}`.

4. **Per-env camera tuning** — warehouses have shelving 4-6 m above the floor; the default `camera_position = (1.2, 1.2, 0.8)` works for the open warehouse but may clip walls in `office`. Two options:
   - **Simple**: same camera, accept that some scenes look cramped.
   - **Right**: per-env camera presets in `ENV_URLS` entries (`camera_position`, `look_at` overrides).
   - Recommendation: ship presets only after smoke-testing each env.

5. **Schema addition** to `meta.json`:
   ```jsonc
   "scene": {
     "environment": {"name": "warehouse", "url": ".../warehouse.usd"},
     // ... existing static_assets, scenario_specs, ground, hdri
   }
   ```
   Backwards-compatible additive change.

6. **`ClipConfig.environments`** defaults to all 6 + `"none"` with uniform sampling. Future iteration: weighted (e.g. warehouse 0.4, room 0.2, office 0.2, none 0.2).

**Outcome:** clips now happen *inside* a warehouse / room / office. The
encoder learns that bodies appear against rich backgrounds with realistic
lighting and occlusion — the biggest sim→real gap from the critique
discussion.

### Tier 3 — Dynamic asset bodies *(1-2 days)*

1. Refactor `asset_library.py` to the category-keyed taxonomy (§4).
2. Expand `CANDIDATE_DYNAMIC_PROPS` to YCB subset + Blocks + Mugs (~40 paths).
3. Add `body_factory.build_asset_bodies(world, rng, config, n_bodies, ...)` parallel to `build_random_bodies` but spawning via `asset_loader.load_dynamic_asset`. Body specs get `asset_url`, `asset_category`; `dims`/`color_rgb`/`roughness`/`metallic` become `null` (asset ships its own).
4. Add `asset_body_fraction: float = 0.5` to `ClipConfig`. In `build_random_bodies`, sample per-body: with probability `asset_body_fraction`, route to `build_asset_bodies` instead.
5. Trust authored masses for v0.1 (plan §9 q3).

**Outcome:** dynamic bodies include cracker boxes, mustard bottles, mugs,
drills, blocks. The encoder sees realistic geometry, not just primitive shapes.

### Tier 4 — Per-scenario asset substitution + jitter *(1 day)*

1. `drop_on_platform`: when `use_asset_platform=True` (config knob), replace primitive platform with a sampled large prop (pallet, sektion_cabinet).
2. `settle`: when `use_asset_bowl=True`, replace primitive bowl with sampled KLT bin.
3. Per-instance scale jitter (log-uniform ±15%) on asset spawn — record `asset_scale_jitter` in body/scene specs.

**Outcome:** scenes show bodies settling inside real KLT bins, dropping
onto pallets. Variety hides the finite asset pool.

---

## 6. Schema additions (all additive)

```jsonc
// meta.json
{
  "scene": {
    "environment": {"name": "warehouse", "url": "..."},      // NEW
    "static_assets": [
      {"prim_path": "...", "asset_url": "...", "category": "boxes_containers"}  // category NEW
    ],
    // existing fields unchanged
  },
  "bodies": [
    {
      "id": 1,
      "asset_url": ".../003_cracker_box.usd",      // null for primitives
      "asset_category": "ycb_food",                // null for primitives  -- NEW
      "asset_scale_jitter": 1.07,                  // null for primitives  -- NEW
      // existing fields unchanged
    }
  ]
}
```

No field renames. `schema_version` stays `"rigid-probe/v0-draft"` until the
shard corpus is generated.

---

## 7. Open questions

1. **Camera placement per env.** A warehouse aisle is open; an office is
   cluttered. Should we (a) keep a single global camera and accept
   variance, (b) per-env presets, or (c) per-clip random camera positions
   constrained to the env's free space? (a) is fastest; (b) is right;
   (c) is overengineering for v0.1.
2. **Env load time.** A full warehouse USD is ~500 MB referencing
   hundreds of textures. Cold-loading per clip could add 10-30 s. Mitigation
   options: (a) accept it for v0.1 corpus generation (one-time cost), (b)
   keep a hot stage between clips of the same env — breaks the per-clip
   isolation, may leak state. Recommend (a) for v0.1, revisit if generation
   throughput becomes the bottleneck.
3. **Env lights interacting with HDRI.** Each env ships authored lights.
   If `config.hdri_paths` is non-empty AND env is loaded, lights compose
   and the scene over-brightens. Recommendation: HDRI takes precedence
   over env lights only when `env == "none"`; otherwise skip HDRI dome.
4. **Body placement inside envs.** The current `(-0.3, 0.3)` body footprint
   places bodies at world origin. Warehouses center their floor at origin;
   office places the door at origin (bodies may end up *inside* a desk).
   Fix: per-env recommended body anchor in `ENV_URLS`, defaulting to origin.
5. **Forklift / shelves variants.** `warehouse_with_forklifts.usd` and
   `warehouse_multiple_shelves.usd` are visually distinct from the plain
   `warehouse.usd`. Treat as separate environments (4 warehouse variants)
   or collapse to one "warehouse" category with the loader picking randomly
   per clip? Treat as separate — gives the encoder more visual diversity.
6. **Asset mass override.** Plan §9 q3 says "trust shipped masses for
   v0.1" but YCB authored masses may be unrealistic for small bodies
   (e.g. foam_brick claims 30 g but is rendered as a 5 cm cube). Audit
   the YCB pool and override outliers per-body before generation.

---

## 8. Relation to existing plans

- **[[scene-diversification]]** §4 (asset library): this plan extends with environment backdrops as a new axis. No conflicts; only additions to `asset_library.py` and new `environments.py`.
- **[[probe-dataset]]**: per-clip schema unchanged; new fields are additive under existing `bodies[i]` and `scene` dicts.
- **[[critique2]]**: directly addresses the critic's "Phase 0.5 visual distribution is monotonous" point — environment backdrops + diverse assets are the biggest lever for sim→real transferability.

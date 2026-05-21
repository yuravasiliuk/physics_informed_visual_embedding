# Synthetic data collection — Isaac Sim pipeline

**Date:** 2026-05-21. Status: pre-implementation engineering plan. Companion to [training-data.md](training-data.md) and [domain-marginalization-regularizer.md](domain-marginalization-regularizer.md).

This document specifies *how* to actually produce the sim half of the corpus described in [training-data.md](training-data.md). The high-level choices (Isaac Lab, ≥30 % free objects, 16-frame clips @ 30 Hz, all seven heads logged per frame) are inherited from that file.

## Decisions fixed in this round

- **Scope.** Both free-object and manipulation. Free-object scenes first (no controller, fastest debug loop); manipulation layered in once the pipeline is stable.
- **Compute.** Single workstation, 1–2 GPUs. Realistic ceiling is **~10k–50k clips total**, not the 100k–500k target in [training-data.md](training-data.md#targets-placeholder--revise-after-pilot). Treat that target as the cluster-scale figure; this collection is the workstation pilot.
- **Realism.** Decent assets + heavy domain randomization, not strict photorealism. Render in Isaac's **RTX Real-Time** mode (or PT at low samples, 4–16) — quality budget goes to scene/material/lighting diversity, not per-frame fidelity. Appearance bridging via Cosmos-Transfer2.5 is deferred to a later pass (option B/C in [training-data.md §Open design decisions](training-data.md#open-design-decisions-resolve-before-pipeline-build-out) point 2).

## Asset sources (decent-tier, free or research-licensed)

### Rigid objects (free-object scenes + manipulation targets)
- **Google Scanned Objects (GSO)** — ~1k photogrammetry-scanned household items, clean meshes, CC-BY. The default. Mass/material reasonable out of the box.
- **Amazon Berkeley Objects (ABO)** — ~8k household items, photogrammetry-grade. CC-BY-4.0.
- **YCB Object Set** — ~80 standard manipulation objects. Mesh quality varies; useful for matching robotics-paper baselines.
- **Objaverse-XL filtered subset** — millions available, but most are unusable raw. Filter: watertight, single-object, scale-annotated, < 50k triangles. Plan ~5–10k items after filtering. Use [Objaverse-XL annotations](https://objaverse.allenai.org/) for category and physical-plausibility filtering.
- **NVIDIA SimReady Asset Library** — small (~500 items) but Isaac-native USD with MDL materials and physics presets. Use for hero objects and known-good props.

### Articulated assets (manipulation phase)
- **PartNet-Mobility (SAPIEN)** — ~2k articulated objects (drawers, doors, scissors, laptops). URDF-ready.
- **Isaac Lab built-in props** — shelves, tables, bins. USD-native, fastest path.

### Robots (manipulation phase)
- **Franka Panda** — Isaac Lab built-in, default. Matches DROID on the real side for symmetric comparison.
- **UR5 / UR10e** — built-in; mirror Open X-Embodiment subset.
- **AgiBot Genie URDF** — if released and stable; matches AgiBot-World-Beta exactly. Otherwise skip in pilot.

### Materials and lighting (the actual realism budget)
- **Poly Haven HDRIs** — 700+ HDR environment maps, CC0. Use for image-based lighting *and* background plates.
- **Poly Haven textures / ambientCG** — ~2k PBR materials, CC0. The core DR pool for randomized rebinding.
- **NVIDIA vMaterials 2** — MDL library shipped with Omniverse; physically correct, free.

### Scenes / backgrounds
- **Isaac Sim sample environments** (warehouses, simple rooms) for grounded shots.
- HDRI dome lighting as primary; explicit room geometry only when occlusion/contact with walls is needed.

## Scene templates

### Free-object scenes (Phase 1)
Target ratio inside the free-object half: roughly equal across these five templates so contact, ω, and v are densely supervised.

1. **Drop** — 1–6 objects spawned at random height above a surface, released with random initial pose and small angular velocity. Densest signal for `v`, `ω`, contact onset.
2. **Stack collapse** — Pre-built unstable stacks (cubes, cylinders) released. Multi-body contact, persistent forces.
3. **Slope roll** — Inclined plane (10°–40°), objects placed at top, released. Clean ω signal, predictable contact normal.
4. **Domino chain** — 5–20 dominoes, one tipped. Sequential contact events, small CoM motion until tip.
5. **Cluttered surface** — 5–15 objects scattered on a tabletop, no manipulation; static frames with occasional perturbation from a "ghost" force. Source of low-motion, contact-zero negatives.

### Manipulation scenes (Phase 2)
1. **Tabletop pick-and-place** — Franka picks one of N (5–20) objects from a tray, places on target. Mirrors AgiBot/DROID task distribution.
2. **Push / topple** — Franka end-effector pushes objects; no grasping. Cheaper, denser contact.
3. **Drawer / cabinet** — PartNet-Mobility articulated targets. Adds articulated contact and non-rigid-body-only joint state.

## Per-frame logged outputs

For each frame in each clip, log:

| Modality | Source | Format |
|---|---|---|
| RGB | Replicator `rgb` annotator | uint8 PNG, 320×320 (downsampled from render res) |
| Depth (camera-frame distance) | `distance_to_camera` | float16 NPZ |
| Normals (camera-frame) | `normals` | float16 NPZ, 3-channel |
| Instance segmentation | `instance_segmentation_fast` | uint16 PNG + id-to-prim mapping JSON |
| Semantic segmentation | `semantic_segmentation` | uint8 PNG + class table |
| Per-body state | custom callback over `RigidObject.data` and `ArticulationData` | Parquet, one row per (body, frame) |
| Contact reports | `ContactSensor` per dynamic body | Parquet, sparse rows per contact event |
| Camera intrinsics + extrinsics | Replicator camera state | JSON per clip |

Per-body state columns: `body_id, frame, mass, CoM_world (xyz), v_world (xyz), ω_world (xyz), pose_world (7-vec), distance_to_ground, kinematic_flag, bbox_diag, compact_flag` (the compact_flag implements the CoM filter from [training-data.md "Label semantics"](training-data.md#label-semantics)).

Camera-frame conversion (CoM, v, ω → camera frame to match [training-data.md "Label semantics"](training-data.md#label-semantics)) happens *offline* at dataloader time so the raw log keeps world-frame state — cheaper to recompute than to re-render if the convention changes.

## Domain randomization knobs (heavy)

Randomized **per clip**, not per frame (frame-coherent within a clip):

- Lighting: HDRI swap, dome rotation, exposure ±2 EV, optional 1–3 area lights at random pose.
- Materials: rebind every object's material from the Poly Haven / ambientCG / vMaterials pool, with per-instance roughness/metallic/normal-strength jitter.
- Camera: FOV ∈ [40°, 80°], pose orbited on a hemisphere around scene center, height ∈ [0.3, 1.8] m, slight roll jitter.
- Object scale: ±20 % per object (with mass rescaled by volume to keep density realistic).
- Object physics: friction ∈ [0.2, 1.2], restitution ∈ [0.0, 0.5], mass ±30 % from default.
- Render mode: random toggle between RTX Real-Time and PT-low (4 spp) so the encoder sees both renderer aesthetics.
- Background: HDRI-only vs HDRI + simple room geometry, 50/50.
- Post: gaussian noise σ ∈ [0, 0.01], optional motion blur, JPEG re-encoding at quality ∈ [70, 95] for ~10 % of clips (matches real-data compression artifacts).

The point is **breadth, not fidelity** — see [domain-marginalization-regularizer.md "Why this exists"](domain-marginalization-regularizer.md). The regularizer assumes the encoder can't trivially classify sim vs real from pixel statistics; aggressive DR is what makes that assumption defensible without the heavy Cosmos-Transfer step.

## Output format and storage

- **WebDataset** (`.tar` shards, ~1 GB each). Each sample = one clip directory containing all per-frame files + state.parquet + meta.json.
- 16 frames × 320×320 × (RGB+depth+normals+seg+state) ≈ 4–6 MB / clip uncompressed; ~2 MB compressed. 50k clips ≈ 100 GB. Fits a single NVMe drive comfortably.
- Manifest table (Parquet): `clip_id, template, n_bodies, contact_event_count, scene_seed, asset_ids, render_mode, dr_seed`. Used to subset and stratify training batches.

## Compute budget and throughput estimate

Working assumptions, RTX 4090-class GPU:

- Isaac Lab vectorized envs: 64–256 free-object envs/GPU at 30 Hz physics; rendering throttles to ~10–30 envs/GPU with one 320×320 camera each at RTX Real-Time.
- Per-clip wall-clock (16 frames + 0.5 s settle): ~3–6 s amortized at 16 parallel envs.
- 10k clips ≈ 8–17 hours single-GPU. 50k clips ≈ 2–4 days.
- Manipulation roughly 3–5× slower per clip due to controller stepping and longer episodes.

Plan for 10k free-object + 5k manipulation as the **workstation-feasible Phase-1 corpus**. Extend to 50k total only if Phase-1 training shows promise.

## Implementation order

1. **Smoke test, 1 clip.** One drop scene, Franka not yet loaded. Verify Replicator emits RGB/depth/normals/seg correctly, custom callback writes per-body Parquet, manifest entry created. Manual inspection of every output.
2. **Sanity dashboard.** Streamlit / matplotlib script: depth histogram, normals unit-norm check, seg mask coverage, contact-event frequency per template, mass/CoM/ω distributions. Run on every batch of new clips.
3. **Phase 1a — 100-clip pilot.** Five free-object templates, full DR on. Check dataloader can read WebDataset shards into training format. Train a 1-epoch B-config smoke run end-to-end.
4. **Phase 1b — 1k clips.** Stratification correct? CoM filter (`compact_flag`) excluding the right bodies? Training loss curves sane?
5. **Phase 1c — 10k clips free-object.** Full Phase-1 corpus. Train B and F to convergence; first real B-vs-F comparison.
6. **Phase 2a — Franka pick-place smoke test.** 100 clips. Verify controller, contact reports under grasp, articulation state logging.
7. **Phase 2b — 5k manipulation clips.** Three manipulation templates. Integrate into Phase-1 corpus.
8. **Decision gate.** If F shows signal over B, push to cluster for 100k+ extension (see [training-data.md §Targets](training-data.md#targets-placeholder--revise-after-pilot)). If not, diagnose using [domain-marginalization-regularizer.md "Diagnostics"](domain-marginalization-regularizer.md#diagnostics--is-the-regularizer-actually-working).

## Open questions

- **Cosmos-Transfer pass — when?** Skipped in Phase 1. After Phase 1c, run Cosmos-Transfer2.5 on a 1k-clip subsample to estimate cost; decide whether to translate the full corpus or only the manipulation slice (where appearance gap to AgiBot/DROID matters most).
- **Differentiable rendering deferred.** Required for [theoretical-paper-plan.md §4](theoretical-paper-plan.md). Genesis or Mitsuba 3 path is a separate effort not in this pilot.
- **Asset license audit.** Objaverse-XL is mixed-license — needs a license filter on the filtered subset before any release/publication.
- **Mass / friction priors from GSO/ABO?** Both ship measured masses for some objects, defaults for others. Decide whether to honor measured values (smaller DR range, more realistic) or override with the DR distribution (broader coverage). Defaults: honor measured when present.

## Connections to the rest of the plan

- This pipeline produces the sim half referenced in [training-data.md §Synthetic data generation](training-data.md#synthetic-data-generation). Real-side ingestion (AgiBot-World-Beta + ScanNet++) is out of scope here.
- The "decent + heavy DR" choice (rather than strict photorealism or Cosmos-Transfer up front) is **only defensible if** the regularizer in [domain-marginalization-regularizer.md](domain-marginalization-regularizer.md) actually bites. The diagnostic on domain-classifier accuracy is the first thing to check after Phase 1c.
- Phase-1c output is the input to the first real B-vs-F experiment (see [training-data.md §Experimental configurations](training-data.md#experimental-configurations-baseline-vs-full)).

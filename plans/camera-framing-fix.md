# Camera framing & visibility fix

Status: **implemented, awaiting Isaac validation** (2026-05-31). Owner: Yura.
Supersedes the static-rig behaviour described in `scene-diversification.md` §10.2.

### As-built notes (deviations / clarifications from the plan below)

- **Pre-roll** → `preroll_trajectory` in [collect_clip.py](../src/data_gen/collect_clip.py);
  framing helpers (`framing_distance`, `rig_angle_centers`, `sample_camera_pose`)
  + the seg-only render-probe (`probe_body_coverage`, `body_seg_coverage`) in
  [clip_capture.py](../src/data_gen/clip_capture.py). The old `sample_camera_rig`
  is gone.
- **Re-teleport before *every* probe** (`_teleport_bodies`, velocities zeroed):
  `rep.orchestrator.step()` can advance physics a frame, so re-posing per probe
  keeps every candidate camera seeing the same max-spread snapshot.
- **`collect_clip` signature changed** to `(base_out, start_idx, config, seed,
  env_override)` and now returns `[(out_dir, ok, reason), …]`; it creates the
  surviving cameras' dirs itself. A rollout with no viewable camera returns `[]`
  (zero clips). The `collect_clip.py` CLI flags are unchanged, so the §8
  acceptance command still applies.
- **Probe render products** are created and `rp.destroy()`-ed per candidate
  (best-effort; failure is caught). If a Kit version lacks `destroy()`, stale
  probe RPs would render during capture (wasted compute, not wrong output) —
  watch for this in the Isaac run.
- The render-probe-can-integrate-physics point means the *probe* sees a pose one
  step off the snapshot; negligible and bounded by the per-probe re-teleport.

## 1. Problem

A 42-clip `throw` smoke run (`~/data/smoke_throw`, 6 rollouts × 7 rig cameras,
`warehouse_shelves` env) is too noisy to train on. The dynamic bodies — the
*only* thing the physics probe is supposed to learn from — are tiny, frequently
occluded, and sometimes entirely absent from frame.

Measured per-clip peak coverage (max over 32 frames of the pixel fraction
belonging to *all* `body_NN` seg classes combined):

| Symptom | Count | Evidence |
|---|---|---|
| Action **fully invisible** (0 body pixels in any frame) | **6 / 42 (14%)** | clips 6, 17, 20, 23, 25, 26 — seg map has no `body_NN` class |
| **Black** frames | 2 / 42 | clips 6, 20 — mean luma ≈ 16, near-empty FOV |
| Only a **subset** of bodies visible | many | clip 16: 1/3; clip 22: 1/4; clip 24: 2/4; clip 27: 1/4 @ 0.04% |
| Bodies **tiny** even when visible | ~all | median peak coverage **~1.2%**; only 2 clips exceed 9% |

## 2. Root causes

All in the camera rig + scene composition:

1. **Environment-blind rig.** [`sample_camera_rig`](../src/data_gen/clip_capture.py#L65)
   drops `n_cameras` viewpoints on a fixed geometric shell (radius 2.8–3.6 m,
   elevation 20–55°) around a fixed look-at, with no knowledge of warehouse
   occluders (shelves, bins, carts, pillars). Cameras land behind props
   (occlusion) or facing a wall / open void (black).
2. **look_at pinned to the floor** at `camera_look_at = (0, 0, 0.1)`
   ([config.py:35](../src/data_gen/config.py#L35)) while bodies live at
   z ≈ 0.9–1.2 and move. Framing centers on the floor, not the action.
3. **Radius fixed and far** (~3 m) for 4–15 cm bodies in a warehouse-scaled
   space → bodies are ~1% of the frame regardless of viewpoint.
4. **No visibility gating.** [`validate_clip`](../src/data_gen/collect_clip.py#L216)
   only checks `seg.max() == 0` (any foreground) — in an env scene the floor/walls
   are foreground, so a clip with zero bodies still passes as `ok`. Empty/black
   cameras are written verbatim.

## 3. Chosen design (locked 2026-05-31)

- **Framing:** pre-roll trajectory pass — run the deterministic physics once with
  **no rendering**, capture the bodies' swept bounding box over all frames, aim
  look_at at its centroid and set radius so the cloud fills a target fraction of
  the frame.
- **Visibility:** render-probe coverage — render one probe frame per candidate
  camera, require body-seg coverage above a threshold, resample if not.
- **Env/scale:** scale bodies up when a large environment is loaded so they read
  at warehouse scale.

## 4. Camera intrinsics (for the framing math)

From [`make_annotators`](../src/data_gen/clip_capture.py#L38): `focal_length=18.0`,
`horizontal_aperture=20.955`, square sensor (1024²). →
HFOV = 2·atan(20.955 / (2·18)) ≈ **60.4°**, half-angle ≈ 30.2°, `tan ≈ 0.582`.

Framing distance for a cloud of radius `Rc` to fill fraction `f` of the half-frame:

```
d = Rc / (f · tan(HFOV/2)) = Rc / (f · 0.582)
```

e.g. `Rc = 0.3 m`, `f = 0.4` → `d ≈ 1.3 m` (vs. the current fixed ~3 m).

## 5. Implementation plan

### Stage 1 — Frame on the body cloud (pre-roll pass)

In [`collect_clip`](../src/data_gen/collect_clip.py#L257), between `build_scene`
and `sample_camera_rig`:

1. Factor the physics-stepping out of [`simulate_clip`](../src/data_gen/collect_clip.py#L155)
   into a `preroll_trajectory(world, bodies, initial_velocities, config)` that:
   `world.reset()` → set velocities → for `frame_count` frames `world.step(render=False)`
   (no `rep.orchestrator.step`, no `capture`) → collect each body's
   `get_world_pose()` + `half_extents`. Returns per-frame body positions.
2. Compute the **swept AABB** of body centroids over all frames; `centroid` =
   AABB center; `Rc` = max over frames/bodies of `||pos − centroid|| + body_bound`.
3. Pass `centroid` + `Rc` into the rig sampler (below). Then **rewind**:
   `world.reset()` + re-set velocities is already done at the top of the real
   `simulate_clip`, so the pre-roll leaves no residue (reset restores authored
   initial transforms; PhysX is deterministic per seed).
4. Cost: `frame_count` non-rendered physics steps — cheap vs. the render loop.

### Stage 2 — Body-cloud-anchored, env-aware rig with render-probe gating

Rework [`sample_camera_rig`](../src/data_gen/clip_capture.py#L65) →
`sample_camera_rig(rng, config, centroid, cloud_radius)`:

- `look_at = centroid` (+ small jitter), replacing the fixed `camera_look_at` base.
- `d = clamp(cloud_radius / (config.frame_fill_frac · tan(HFOV/2)),
  config.camera_radius_clamp)` — replaces the fixed `camera_radius_range`.
- Keep the evenly-spaced-azimuth + per-rollout global rotation + elevation-band
  spread, all relative to `centroid`.

Then **render-probe gate** each candidate (in `collect_clip`, after the rig is
proposed but before the real capture loop):

- For each camera: create its annotator, render one probe frame
  (`rep.orchestrator.step()` once), compute body-seg coverage =
  `isin(seg, body_ids).mean()`.
- Probe at the **max-spread frame**: after `world.reset()`, teleport bodies to
  their pre-roll snapshot at the frame where the cloud AABB is largest
  (`set_world_pose`), probe, then reset for the real run. This catches occlusion
  at the worst moment, not just t=0. (Default; if teleport proves fiddly, fall
  back to probing the post-reset t=0 pose — documented trade-off.)
- If coverage < `config.min_body_coverage`, **resample** that camera's
  azimuth/elevation (and nudge radius in) up to `config.camera_resample_attempts`.
- If still failing, **drop** the camera. `n_cameras` becomes a *target*: resample
  toward it, accept fewer if a rollout is genuinely hard to view. Clip dirs are
  allocated from the cameras that survive.

### Stage 3 — Scale bodies up in env scenes (+ auto-size the scenery to match)

[`build_scene`](../src/data_gen/collect_clip.py#L120) already receives `env_loaded`.
When `env_loaded`, build with an up-scaled config before dispatching to
`scenarios.build`:

```python
if env_loaded:
    config = dataclasses.replace(config, scale_range=config.scale_range_env)
```

This flows through `config.scale_range` →
[`sample_shape_dims`](../src/data_gen/body_factory.py#L39) for every scenario.

**Footprint blowout (found in the first Isaac run):** enlarging the bodies alone
makes them overflow every scenario's fixed spawn box and static geometry —
`throw` could not place 6 bodies of radius ~0.39 m in the 0.6 m spawn box →
`RuntimeError`. Fix (chosen: *auto-size every footprint to body radius*): a
**uniform scene-scale factor**
[`_scene_scale`](../src/data_gen/scenarios.py) = `max(1, scale_range[1] /
BODY_SCALE_REFERENCE)` (= 1.0 at the tuned reference, so non-env scenes are
unchanged). Every spawn box and procedural asset (ramp, platform, bowl, topple
footprint/height/centers, collide half-sep/height) is multiplied by it via
`_scaled(...)`, so the apparatus keeps the *same relative proportions* as the
bodies grow. `BODY_SCALE_REFERENCE = 0.15` is recorded in
[config.py](../src/data_gen/config.py); the per-clip factor is written to
`meta.json["scene"]["scenario_specs"]["scene_scale"]`.

Topple now scales too (its `_sample_tall_dims` takes the scaled
footprint/height ranges) — resolving the earlier "topple ignores scale" caveat.
Cluttered's real static props are fixed-size, so only its dynamic-body spawn box
scales. Verified off-Isaac: 0/300 placement failures and 0 collide spawn
overlaps at both base (s=1.0) and env (s=3.0) scale.

### Stage 4 — Harden `validate_clip` (write-time safety net)

Replace the `seg.max() == 0` check in
[`validate_clip`](../src/data_gen/collect_clip.py#L233) with a **body-coverage**
check: require peak `isin(seg, body_ids)` coverage ≥ `config.min_body_coverage`
(same threshold as the probe). This stops env-scene clips with zero/near-zero
bodies from being marked `ok`, independent of the rig changes — and is the last
line of defence if a probed camera still degrades over the rollout.

## 6. Config additions ([config.py](../src/data_gen/config.py#L37))

```python
# Replaces camera_radius_range + the fixed look_at base for the rig.
frame_fill_frac: float = 0.4          # target cloud fraction of half-frame
camera_radius_clamp: tuple = (0.8, 4.0)   # m, min/max framing distance
min_body_coverage: float = 0.01       # reject cameras below this body-pixel frac
camera_resample_attempts: int = 8     # az/elev retries before dropping a camera
scale_range_env: tuple = (0.15, 0.45) # bodies in large envs (vs scale_range 0.05–0.15)
```

`camera_look_at` / `camera_position` stay as the single-camera verify fallback.
`camera_radius_range` can be retired once the rig no longer reads it.

## 7. Risks / open questions

- **Pre-roll determinism:** relies on `world.reset()` + re-applied velocities
  reproducing the rendered run exactly. True today (the real run does the same),
  but verify the pre-roll and real trajectories match to confirm no
  render-vs-no-render divergence.
- **Probe cost:** one render per candidate camera × resamples. Worst case
  `n_cameras · resample_attempts` renders per rollout. **Resolved (2026-05-31):**
  cost is acceptable — quality beats throughput. No raycast pre-filter needed up
  front; add only if it becomes painful.
- **Variable clip count:** a rollout may yield < `n_cameras` clips.
  **Resolved (2026-05-31):** fine — quality over quantity, drop marginal cameras
  freely. Downstream tooling must already tolerate per-rollout clip counts; do not
  pad to hit a target.
- **Teleport-probe fidelity:** `set_world_pose` to a pre-roll snapshot ignores
  any pose the renderer would interpolate; acceptable for a coverage probe.
- **scale_range_env values** are a first guess — tune against a warehouse smoke run.

## 8. Acceptance test

Re-run the smoke set (Isaac, user-run):

```
${ISAAC_SIM}/python.sh src/data_gen/collect_clip.py --result ~/data/smoke_v0.3 \
    --scenario throw --n_clips 6 --seed 0
```

Then the coverage script (this plan's §1 method). Target:
- **0** clips with zero body pixels.
- **0** black frames.
- median peak body coverage **≥ ~10%** (vs. ~1.2% today).
- every surviving camera sees ≥ `min_body_coverage` and the full body set is
  represented across the rig.

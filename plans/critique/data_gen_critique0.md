# Data Generation Critique — Round 0

Critique of `src/data_gen/collect_clip.py` and `src/data_gen/verify_collect_clip.py`
(with supporting analysis of `clip_capture.py`, `scenarios.py`, `body_factory.py`,
`environments.py`, `lighting.py`, `static_geometry.py`, `seg_labels.py`,
`asset_library.py`, `asset_loader.py`).

The student should fix these himself. This document is the issue list, not the patch.

---

## Critical Issues

### 1. RGB saved only as lossy MP4 — no raw frames preserved

In `clip_capture.py:save_clip()`, RGB is written only via `cv2.VideoWriter` with
the `mp4v` codec (MPEG-4 Part 2). Depth, normals, and seg are saved losslessly
as `.npy`, but RGB is not.

For linear probing experiments where you're training decoders on ground-truth
physics quantities aligned with per-frame visual features, the encoder receives
lossy-compressed input while labels come from exact simulation. This injects
codec-specific noise into your gradients. Particularly concerning at 1024×1024
where `mp4v` can produce visible blocking artifacts.

**Why it matters for the experiment:** if you train V-JEPA 2 linear probes on
mp4-decoded frames and compare against exact simulation labels, the codec noise
will artificially degrade your R². You might wrongly conclude "the encoder
doesn't encode physics" when it's actually "the codec destroyed the signal the
encoder needed."

**Fix direction:** also save `rgb.npy` (uint8, lossless), or switch to H.264
CRF-0 (lossless) or PNG sequences. Keep the mp4 for human inspection.

---

### 2. Initial velocities are not persisted anywhere

In `collect_clip.py:collect_clip()`, after `build_scene()` returns
`initial_velocities`, these are applied to bodies inside `simulate_clip` and
then discarded. They don't appear in `meta.json`, `body_specs`, or `state.json`.

The first frame of `state.json` records velocity AFTER one `world.step()`
(one render_dt = 1/30s, 2 physics substeps at 1/60s each). For a thrown body:
`v_logged_frame0 = v_initial + g * dt ≈ v_initial + (0, 0, -0.327)`. You can
approximately back-compute, but contacts/collisions in the first step make this
unreliable.

For the `throw` and `push` scenarios in `scenarios.py`, the per-body velocity
vectors are sampled randomly inside the scenario builder and permanently lost
after application. This breaks:

- Exact reproducibility from `meta.json` alone (only the sampling *range* is
  recorded under `scenario_specs`, not the realized sample)
- Any training target that requires initial conditions
- Any debugging path where you want to know "what did this clip start from?"

**Fix direction:** store `initial_velocities` per body in `body_specs` (as
`init_lin_vel_world` / `init_ang_vel_world`) or as a top-level `meta.json` field.

---

### 3. No retry logic for validation failures in batch generation

`collect_clip.py:main()` calls `collect_clip()` once per seed and accepts the
result. When `--n_clips=1000`, each failed clip
(`all_bodies_stationary`, `empty_seg`, `rgb_low_variance`, `nonfinite_state`,
etc.) wastes one Isaac Sim iteration (~2-5 seconds at this resolution) and
writes a useless clip directory to disk.

The plan documents say "the shard driver can re-seed on failure" but this is
unimplemented. With 7 scenarios and random spawning, failure rates can be
5-20% — especially `settle` and `cluttered` where bodies may not move enough,
or where rejection sampling fails.

The dataset will be polluted with clips marked `validation.ok=False`. Any
downstream dataloader must filter on this — easy to forget.

**Fix direction:** wrap the call in a bounded retry loop, e.g. up to 3
attempts with seeds `seed`, `seed + 1_000_000`, `seed + 2_000_000`. Skip writing
to disk on failure (or write to a `failed/` subdirectory) so the main tree
contains only valid clips.

---

## High-Priority Issues

### 4. Verification script only tests the simplest possible case

`verify_collect_clip.py` validates ONLY:

- Single sphere (no multi-body index confusion)
- Zero angular velocity throughout the clip
- Free flight after a single linear impulse (no contact, no friction during the
  airborne window that's actually checked)
- Spherical geometry, where `local_com == 0` so the
  `pos_world + R_body_world @ com_local` math in `log_state_frame` is trivially
  satisfied

It does NOT test:

- **Angular velocity logging correctness.** The sphere has ω=0 throughout; a
  sign flip in `R_camera_world @ ang_vel_world` would pass silently.
- **Quaternion orientation logging.** `quat_camera_wxyz` is computed via
  `quat_wxyz_mul(q_camera_world, quat_body_world)` but never verified against
  an analytical expectation.
- **Non-spherical bodies.** Cuboids/cylinders with asymmetric mass distribution
  exercise the `R_body_world @ com_local` path; this is currently dead code
  from the test's perspective.
- **Multi-body correctness.** Are body indices in `state_records` consistent
  with `body_specs`? Could two bodies be swapped in the records list?
- **The production camera viewpoint** at `(2.2, 2.2, 1.8)`. Verify uses a
  custom broadside camera at `(1.0, -3.0, 1.2)`. A degenerate edge case in
  `world_to_camera_frame` could pass verify and fail in production.

This gives false confidence. You could have a sign error in angular velocity
that only manifests when you train probes on ω and get random performance —
weeks lost.

**Fix direction:** add verification cases for:
- A spinning cuboid (verify ω is constant during free flight, magnitude
  matches launch ω in both world and camera frames)
- A body with known asymmetric CoM offset (e.g., a composite or a non-uniform
  density override) — verify `read_local_com` returns the expected offset and
  the logged `com_camera` matches an analytical world→camera computation
- A two-body scene — verify per-frame records for body A and body B are not
  swapped after Isaac's `world.scene` internal ordering kicks in
- Run verify with the *production* camera position as well as the broadside
  one

---

### 5. `seg_info` captured only from frame 0

In `clip_capture.py:save_clip()`:

```python
seg_info = frames[0].get("seg_info")
if seg_info is not None:
    with open(... "seg_id_to_class.json") ...
```

If a body is fully occluded in frame 0 but becomes visible in later frames, its
class→ID mapping might be absent from frame 0's `seg_info`. Behavior depends on
the Replicator version:

- Some versions report all stage-authored semantics regardless of visibility
- Others report only visible labels per-frame

If the latter, your `seg_id_to_class.json` is incomplete. Downstream per-body
pooling code will then silently mis-route segmentation IDs to "unknown" for
some frames.

**Fix direction:** merge `seg_info` across all frames before saving:

```python
merged = {}
for f in frames:
    info = f.get("seg_info")
    if info:
        merged.update(info)  # last write wins is fine — IDs are stable across frames
```

Then verify empirically (one of the verify-script extensions) that semantic IDs
stay stable across frames for the same class.

---

### 6. Segmentation stored as uint32 — massive disk waste

In `clip_capture.py:save_clip()`:

```python
seg_stack = np.stack([f["seg"] for f in frames])  # dtype=uint32
```

At 1024 × 1024 × 32 frames × 4 bytes = **128 MB per clip** for segmentation
alone. For a 10k-clip dataset, that's **1.28 TB** just for seg masks.

The scene contains at most ~10 semantic classes (≤8 dynamic bodies + scenery +
background). uint8 suffices (max 255 distinct IDs). That's **32 MB per clip**
— 4× savings, or ~960 GB freed on a 10k-clip dataset.

The annotator returns uint32 because that's its native output, but downcasting
before save is safe as long as you assert the max ID fits.

**Fix direction:** in `capture()` or `save_clip()`, after asserting
`seg.max() < 256`, cast to uint8. If you ever exceed 255 classes (unlikely),
fall back to uint16 and assert that fits. Don't keep uint32.

---

## Medium-Priority Issues

### 7. No mass/density randomization — mass is trivially predictable from visual size

All bodies use Isaac's default density (1000 kg/m³ for primitives). Mass =
density × volume, and volume is determined by `shape` + `dims` which are
visually observable. A "predict mass from video" probe would just learn to
estimate visible volume — it's not testing whether the encoder represents mass
as an independent physical property.

In real physics, objects of the same visible size can have vastly different
masses (steel ball vs. ping-pong ball, full vs. empty bottle). Without density
variation, the dataset cannot probe for this distinction.

This is a **dataset design limitation**, not a bug — but it directly undermines
the probing-study direction (Direction A in the upstream plan). If you intend
to probe for "does V-JEPA 2 encode mass independent of size," this dataset
cannot answer that question.

**Fix direction:** add `density_range` to `ClipConfig`, sample per body in
`body_factory.build_random_bodies`, set via `UsdPhysics.MassAPI` (or use
`set_mass` on the rigid prim view). Log density per body in `body_specs`.

---

### 8. `mass` logged redundantly every frame in `state.json`

`log_state_frame` writes `"mass": float(body.get_mass())` for each body at each
frame. Mass is constant through the clip. With 8 bodies × 32 frames = 256
redundant float entries per clip. ~7% of `state.json` size is duplicated mass
data. Multiplied by 10k clips, this is real disk and parse-time overhead.

**Fix direction:** move mass to `body_specs` in `meta.json` (a per-body scalar,
written once) and drop it from per-frame records.

---

### 9. `body_specs` in `meta.json` doesn't include mass

Related to #8. Mass is available from `state.json` but not from
`meta.json["bodies"]`. For a dataloader that only needs scene-level metadata,
this forces parsing the full `state.json` (typically 200KB+ of per-frame state)
to extract a single per-body constant. Ergonomics issue, but real.

**Fix direction:** part of the #8 fix — write `mass` once into each entry of
`body_specs`.

---

### 10. Off-by-one: first logged frame is post-step, not initial state

The simulation loop in `simulate_clip` is:

```python
world.reset()
# set initial velocities
for frame_idx in range(K):
    world.step(render=True)
    rep.orchestrator.step()
    frames.append(capture(annots))
    log_state_frame(...)
```

Frame 0 in the output records the state at `t = render_dt`, not `t = 0`. The
verify script correctly accounts for this with
`t = (i - launch_frame + 1) * render_dt`, so it's by design.

But for **training** code that pairs an RGB frame with a velocity label: at
frame 0 you have RGB(t=dt) paired with v(t=dt), and v(t=dt) has already been
modified by gravity (∆v_z ≈ -0.327 m/s for one render step) and possibly by
contacts. The "pure" initial condition is never captured.

This is acceptable for probing (you're predicting current-frame physics, not
initial conditions) and is internally consistent (RGB and state are
co-stepped). It's mentioned here so the student documents it explicitly and
downstream code doesn't assume `state[0]` = initial conditions.

**Fix direction:** either (a) capture a true t=0 frame before the first step
(requires a separate `capture()` call before the loop), or (b) document in the
schema doc that "frame i corresponds to t = (i+1) * render_dt after velocity
application."

---

## Design Observations (Not Bugs, but Worth Flagging)

### 11. Replicator camera vs. hand-coded `world_to_camera_frame` rely on matching conventions

The hand-coded look-at matrix in `clip_capture.py:world_to_camera_frame` must
match Replicator's internal camera matrix exactly. Both currently use
USD/OpenGL convention (+X right, +Y up, -Z forward). The verify script's
ballistic test implicitly validates this — sub-mm reconstruction errors across
the airborne window — but only for one camera orientation. If Replicator ever
changes its internal up-vector convention or focal-length-to-extrinsic
relationship, state labels will drift from images silently.

**Mitigation:** the verify script's roundtrip check is the regression test for
this. Run it on every Isaac version bump.

---

### 12. `world_to_camera_frame` degenerates for top-down cameras

If `camera_position` sits directly above `look_at`
(i.e., `forward ∥ world_up`), then `cross(forward, world_up)` is the zero
vector and `right /= np.linalg.norm(right)` produces NaN. The NaN propagates
into every state record.

Current production camera `(2.2, 2.2, 1.8) → (0.0, 0.0, 0.1)` is safe.
Verify camera `(1.0, -3.0, 1.2) → (1.0, 0.0, 0.4)` is also safe.

But if someone adds a top-down or birds-eye scenario, the system fails
silently (no assertion, no error — just NaN state). The `validate_clip`
`nonfinite_state` check would catch it post-hoc, but only after a full clip's
worth of compute wasted.

**Fix direction:** in `world_to_camera_frame`, assert
`|cross(forward, world_up)| > eps` before dividing. Or accept an alternate
world-up when the camera is near-vertical.

---

### 13. Replicator's internal RNG not seeded

`np.random.default_rng(seed)` controls scene-level randomization, but
Replicator's ray-tracing and any internal denoiser have their own RNG state
that's not seeded through the public API. Clips are not bit-exact reproducible
across runs with the same `--seed`.

This is fine for training data generation but means you cannot reproduce a
single "interesting" clip exactly for debugging.

---

## Summary Table

| #  | Issue                                             | Severity | Why it matters                                    | Fix effort |
|----|---------------------------------------------------|----------|---------------------------------------------------|------------|
| 1  | RGB lossy-only (mp4v)                             | Critical | Codec noise corrupts probe training signal        | Low        |
| 2  | Initial velocities lost                           | Critical | Reproducibility broken; can't be used as labels   | Low        |
| 3  | No retry on validation failure                    | Critical | 5-20% compute wasted; polluted dataset            | Low        |
| 4  | Verify script too narrow (ω, quat, multi-body)    | High     | False confidence in logging correctness           | Medium     |
| 5  | seg_info from frame 0 only                        | High     | Potential missing body IDs in seg mapping         | Low        |
| 6  | Seg stored as uint32                              | High     | 4× disk waste (~1 TB on 10k clips)                | Trivial    |
| 7  | No density randomization                          | Medium   | Mass probe is trivial (mass ∝ visible volume)     | Medium     |
| 8  | Mass logged every frame                           | Medium   | ~7% bloat in state.json                           | Trivial    |
| 9  | Mass missing from `body_specs`                    | Medium   | Forces full state.json parse for constants        | Trivial    |
| 10 | Frame 0 = post-step, not t=0                      | Medium   | Documentation/clarity issue                       | Trivial    |
| 11 | Replicator vs. handcoded camera convention        | Note     | Regression risk on Isaac version bump             | N/A        |
| 12 | `world_to_camera_frame` degenerates top-down      | Note     | Silent NaN if camera config changes               | Trivial    |
| 13 | Replicator RNG not seeded                         | Note     | Single-clip reproducibility broken                | Hard       |

---

## Recommended Fix Order

Before running any training experiments:

1. **#1 (RGB lossless)** — without this, probe results are uninterpretable.
2. **#2 (persist initial velocities)** — needed for any training target that
   uses initial conditions, and for full reproducibility.
3. **#4 (extend verify script)** — must catch ω / quaternion / multi-body bugs
   *before* spending GPU-weeks training on potentially-wrong labels.
4. **#5 (merge seg_info)** — quick win that removes a silent failure mode.
5. **#6 (seg as uint8)** — quick win that saves ~1 TB on a 10k-clip run.
6. **#3 (retry logic)** — needed before large-scale generation runs but not
   blocking for a small first batch.

#7 (density randomization) is a separate decision: include it if mass-as-an-
independent-quantity is a target probe, defer if not.

#8, #9, #10 are cleanup that can ride along with the other meta.json changes.

#11, #12, #13 are watch-items, not action-items.

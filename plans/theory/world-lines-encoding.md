# Encoding sim data → world-lines (Phase E0)

**Status:** implemented; round-trip verified to ~1e-16 on the synthetic generator.
Code: [`src/worldlines/`](../../src/worldlines/). Parent design doc:
[world-lines.md](world-lines.md) (read its §2 frames/units and §6 losses first). This file
is the *encoding* layer only — turning a clip into an inspectable `WorldLines` object.

**Goal of this phase:** a small, dependency-light Python package that turns a collected
clip (or a synthetic stand-in) into an inspectable `WorldLines` object, with every modeling
choice — gauge, units, basis, degree — a tweakable knob. **No model, no training, no GPU.**
This is the "understand exactly what we encode, and be able to change it" layer; everything
runs in a Jupyter notebook on CPU.

## 1. What we read (the sim data contract)
From [../data-gen/probe-dataset.md](../data-gen/probe-dataset.md) /
[clip_capture.py](../../src/data_gen/clip_capture.py):
- `state.json` — flat list of per-body, per-frame records, **all in camera frame**:
  `{frame, body, com_camera, lin_vel_camera, ang_vel_camera, quat_camera_wxyz, mass}`.
- `meta.json` — `R_camera_world` (3×3, rotates a *world* vector into camera coords),
  `camera_position_world`, `render_fps`, `frame_count`, and `bodies` (per-body intrinsics:
  `mass, density, restitution, com_local_authored, dims, shape`).
- Inversion to world frame (the loader's core):
  `p_world = R_cwᵀ · p_cam + cam_pos`; `v_world = R_cwᵀ · v_cam`;
  `R_body_world = R_cwᵀ · R(quat_camera_wxyz)`.
- **Caveat — static camera.** Each clip stores a *single* `R_camera_world`, so the in-clip
  ego-line is **constant**. The moving-camera case (where ego/object disentanglement —
  [world-lines.md](world-lines.md) §7 — actually bites) is **not** in current sim data; it
  comes from the synthetic generator (§5) until/unless data-gen renders intra-clip motion.
- **Visibility** is not in `state.json`; derive `e(t)` from `seg.npz` body-id presence if
  present, else default to all-visible (a hook, not a blocker, for v0).

## 2. The common interface (`RawClip`) — sim and synth share it
A single in-memory structure both the Isaac loader and the synthetic generator produce, so
nothing downstream knows or cares which it came from:
- `RawClip{ fps, times(T,), cam_R_cw(T,3,3), cam_pos_w(T,3), K(3,3), g(3,), bodies[] }`
- `BodyTrack{ id, name, pos_w(T,3), quat_w(T,4 wxyz), lin_vel_w(T,3), ang_vel_w(T,3),
  visible(T,), intrinsics{} }`

All arrays in **metric world frame** — the ground truth before any gauge/units choice.

## 3. The encode pipeline (each step a separate, testable function)
1. **`load_clip(dir) → RawClip`** — read JSON, invert camera→world (§1).
2. **`build_gauge(raw, scheme) → Gauge`** — `scheme ∈ {metric, gravity, scene}`
   (the tiered length gauge of [world-lines.md](world-lines.md) §2.4):
   - `metric`: `R=align(g→−ẑ)` (identity in sim), `t=0`, `L=1`, `T=1`.
   - `gravity` *(default)*: `T=` clip duration (so `τ∈[0,1]`), `L=‖g‖·T²` (⇒ `g≡1`) — tier 1.
   - `scene`: `L=` frame-0 body-span — tier 2 fallback (composition-dependent; ablation only).
   - `Gauge` is **invertible** and stores `R, t, L, T`; points map `p ↦ R(p−t)/L`,
     velocities `↦ v·T/L`, rates `↦ ω·T`.
3. **`to_gauge(raw, gauge) → RawClip`** — apply the gauge to every body + the ego-line. Pure,
   reversible; the only place units/frame change. (Orientation passes through: the fit uses
   `R(0)ᵀR(t)`, which is invariant to a global left-multiply by the gauge rotation.)
4. **`fit_basis(traj, degree) → coeffs, recon_err`** — least-squares fit of the (gauge-frame)
   position to a fixed temporal basis `B(τ)` (Chebyshev, degree 2–4). Orientation via
   `fit_rotations`: fit the **cumulative-unwrapped** `so(3)` tangent (frame-to-frame steps
   summed) so a constant ω is exactly linear and the `π`-wrap of `rotvec` is avoided
   ([world-lines.md](world-lines.md) §11 open-question 1, resolved).
5. **`intrinsics(body) → dict`** — invariants only, R8-clean: `mass_ratio` (mass / clip-mean
   mass), `com_offset_over_size` (‖com_local‖ / body radius — R7), `restitution`, `density`.
   **No appearance fields** (color/roughness deliberately dropped).
6. **`encode_clip(dir|raw, EncodeConfig) → WorldLines`** — orchestrates 1–5 and the ego-line.

`WorldLines{ gauge, ego(EgoLine), lines[WorldLine], meta }`, where
`WorldLine{ id, name, intrinsics, pos_coeffs, q0, rot_coeffs, visible, pos_err, rot_err }`.

## 4. Verification (the round-trip)
We lack 2D GT in sim `state.json`, but we have exact 3D camera-frame state, so the
round-trip is: **reconstruct the world-line from coeffs → re-project to camera frame →
compare against the stored `com_camera`.** `reproject_error(worldlines, raw)` must be
≤ basis-fit tolerance. Plus `fit_basis`/`fit_rotations` checked on analytic curves (a
parabola fits to ~0 at degree 2; constant-ω spin fits to ~0 on `so(3)`). Current synthetic
demo: `pos_err`, `rot_err`, `reproj` all ~1e-16. Run: `python -m worldlines.demo` from `src/`.

## 5. Synthetic generator (`synth.py`) — play without Isaac
Analytic `make_synth_clip(n_bodies, moving_camera, seed) → RawClip`: ballistic translation
`p(t)=p₀+v₀t+½gt²` + constant-ω spin, randomized intrinsics, **optional moving camera**.
This (a) unblocks notebook play before the Isaac P0.4 run, (b) provides exact GT for the §4
round-trip, and (c) is the **only** current source of a non-trivial ego-line (§1 caveat).
Limitation: ballistic+constant-ω is exactly degree-2/linear, so degree≥2 fits it trivially —
it cannot yet stress the basis. Contact/drag stress comes with E1.

## 6. Module layout (`src/worldlines/`, concise, minimal comments)
`io.py` (load_clip, RawClip/BodyTrack) · `gauge.py` (Gauge, build_gauge, to_gauge) ·
`basis.py` (cheb_basis, fit_basis, reconstruct, so3 fit/unwrap) · `encode.py` (WorldLine,
EgoLine, WorldLines, encode_clip) · `project.py` (world→cam→pixel, reproject_error) ·
`synth.py` (make_synth_clip) · `demo.py`. Deps: `numpy`, `scipy` only.

## 7. What is tweakable (the point of this phase)
Gauge scheme & units ([world-lines.md](world-lines.md) §2.4), basis degree, which intrinsics
are emitted (and the R8 exclusion list), the contact-residual on/off
([world-lines.md](world-lines.md) §5 — deferred to E1), visibility source. Each is an
`EncodeConfig` field, changed in one line in a notebook.

## 8. Out of scope for E0 (named, not silently dropped)
Object queries / the frozen-encoder head (E2, model work), set-matching loss, contact
residual `δ_i` (E1), real-data gauge *estimation* from depth/normals (here the sim gauge is
known from GT), metric-scale recovery from a reference object.

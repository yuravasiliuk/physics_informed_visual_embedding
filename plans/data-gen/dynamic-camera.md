# Dynamic camera — v1 "flying camera" spec

**Status (2026-07-27):** trajectory layer **implemented + verified**
([camera_paths.py](../../src/data_gen/camera_paths.py),
[verify_camera_paths.py](../../src/data_gen/verify_camera_paths.py) — 14/14, runs
without Isaac); collection-side wiring (§6 hooks 3–6) not yet built. This is a
**non-retrofittable knob** (it changes pixels *and* label semantics), settled on
paper before any batch. Decision (2026-07-24): **v1 = flying camera**; robot-mounted
cameras (H1/G1) deferred — see §11.

**Superseding decision (2026-07-27): the corpus is regenerated from scratch**, so
the retrofit objection is moot and every knob is in play. Consequence: `static` is
not a separate mode but the **zero-coefficient case of the dynamic
parametrization** — one code path, one label contract, both slices from the same
sampler.

**Consumers:** [probe/physics-lora-adapter.md](../probe/physics-lora-adapter.md) §3/§6
(egomotion twins, gravity-tilt target), and the postponed gravity-tilt decision
recorded there.

---

## 0. Gravity tilt is DEMOTED, not promoted (2026-07-27)

Regeneration + roll fix the *degeneracy*, but not the deeper objection: **tilt is
recoverable from appearance alone, with no physics.** A single frame's horizon,
ground plane, and vertical edges determine it — GeoCalib does exactly this densely
from single images with zero dynamics. Contrast restitution: you cannot estimate it
without watching a bounce. Camera motion raises the bar (a horizon detector must
now *track*) but doesn't change the kind of cue.

→ Keep tilt as an **enabling signal** (every camera-frame physics judgment needs to
know which way is down) and a **sim+real thermometer** (real-side pseudo-labels
exist: IMU, monocular estimators). Do **not** let it carry the physics claim —
restitution, friction and contact events do that. Settling test stays §9.3
(single-frame vs full-clip decodability gap).

## 1. What this unlocks (why it's worth a knob)

1. **Gravity tilt becomes a real target.** Today `g_cam = R_cw @ (0,0,-1)` is
   degenerate: look-at cameras have **zero roll by construction**
   (`up = cross(right, forward)`, [clip_capture.py:234](../../src/data_gen/clip_capture.py)),
   so `g_cam ≡ [0, -cosθ, -sinθ]` — a 1-parameter arc readable off the horizon.
   Roll + motion make it a genuine `(T', 3)` trajectory with 2 free DoF.
2. **Egomotion labels for free** (sim knows them exactly) — and they are the rare
   sim head with **real-side pseudo-labels** (SfM/SLAM), so a better R2 transfer
   thermometer than depth.
3. **Egomotion twins** — same rollout, different camera path: pull-pairs for
   view-*invariant* targets, exact transform-then-match supervision for
   view-*equivariant* ones (the classification table in the LoRA design).
4. **Closer to V-JEPA's pretraining distribution.** Real video is overwhelmingly
   moving-camera; a static rig is an egomotion outlier. Helps the domain story
   independently of any label.

## 2. Scope guard — what this does NOT change

- **Physics is untouched.** The camera is render-side: same seed ⇒ same rollout,
  bit-identical dynamics. Therefore **all existing `com_oracle` certifications
  stay valid** (the oracle reads `state.json` world-recoverable state, not pixels).
- **Existing corpora stay valid** — dynamic-camera clips are a **separate slice**,
  never silently mixed into a corpus whose probes assume static cameras (§5.3).
- Static camera remains the default: a `static` path family reproduces today's
  behaviour exactly (§3), so nothing regresses.

## 3. Trajectory families

A path is a seed-deterministic map `t -> (pos_world, look_at_world, roll_rad)`,
sampled once per camera and evaluated per frame. All families are built around the
**existing** framing solve (`framing_distance`, cloud centroid/radius) so the action
stays in frame:

| Family | Parametrization | Purpose | Tilt coverage (measured) |
|---|---|---|---|
| `static` | all coefficients zero | default; backward-compatible | none — `x ≡ 0` |
| `orbit` | az drift `ω_az ∈ [±2, ±12] deg/s` about the cloud | parallax, smooth | **none** (gauge, see below) |
| `arc` | elevation drift, clamped to `camera_elevation_clamp_deg` | pitch variation | span only, `x ≡ 0` |
| `dolly` | radial in/out, `d(t)` within `camera_radius_clamp` | scale/FoV change | **none** (gauge) |
| `roll` | `roll(t) = A·sin(2πft+φ)`, `A ∈ [5, 25]°`, `f ∈ [0.1, 0.5] Hz` | **the degeneracy breaker** | `|x|max ≈ 0.25`, span 13° |
| `handheld` | sum-of-sinusoid 6-DoF jitter + roll | real-video-like | `|x|max ≈ 0.25`, span 24° |
| `gait` | handheld + `1.5–2.5 Hz` bob + step-synced roll (half rate) | embodied-video spectrum without a robot (§11) | `|x|max ≈ 0.06`, span 5.5° |

**★ The gauge result (measured 2026-07-27, now gated in `verify_camera_paths`).**
The four unobservable DoF — 3 translation + yaw about gravity — are **exactly** the
motions that leave `g_cam` invariant. So `orbit` (pure yaw about the world-Z =
gravity axis) and `dolly` (pure translation) contribute **zero** tilt variation, and
`arc` (pure pitch) varies the label *along the degenerate arc* while keeping
`x ≡ 0` — i.e. it adds tilt variation without adding tilt *information*.
**Only roll breaks `x ≡ 0`.** Consequences for the mix:
- roll is **mandatory** in the dynamic slice, not one family among several — use
  `sample_camera_path(..., roll_always=True)` for any batch that must carry tilt.
- `gait`'s roll is deliberately small (a few degrees is what walking actually
  produces) — keep it honest for realism and get coverage from `roll`/`handheld`
  rather than inflating gait.
- orbit/dolly still earn their place: parallax, scale change, and egomotion
  supervision. They just cannot be counted toward tilt coverage.

Composable: a path is a base family + optional roll and jitter components. All
families share **one** parametrization (az drift · el/dolly/roll sinusoids ·
jitter · bob); a family is a preset of coefficients, and `static` sets them all
to zero. Jitter is a sum of sinusoids, not filtered noise, so `pose(t)` is a pure
function of `t` — no filter state, frames evaluable in any order, paths
re-derivable from `params` alone (this is what keeps twins reproducible, §4).
**Speed bound (acceptance criterion, §9):** egomotion-induced image flow must stay
below object-induced flow for the majority of frames, else the clip is
egomotion-dominated and the physics targets get swamped.

**Verified properties** (`verify_camera_paths.py`, all closed-form, no Isaac):
- `roll_rad=0` reproduces `clip_capture.world_to_camera_frame` **bit-wise** →
  today's behaviour is the zero-coefficient case, provably.
- Static path: `g_cam = [0, −0.819, −0.574]`, constant, `x ≡ 0` — the degeneracy,
  demonstrated. Roll path: `|x|max 0.215`, `std 0.091` — 2-DoF restored.
- **Pure translation (dolly) gives EXACTLY zero tilt variation** — gravity is a
  uniform field, so tilt depends on camera *orientation* only. Budget rotational
  families for tilt coverage; dolly buys parallax, not tilt.
- Temporal equivariance `g(t+1) == R_rel(t)·g(t)` holds to 1e-12; same seed →
  identical path; elevation/radius clamps hold across the whole clip.
- Gait egomotion magnitudes: ~0.18 m/s, ~0.19 rad/s (sanity band for the §9.4 flow bound).

## 4. Determinism + twin discipline (load-bearing)

- **Camera RNG must come from a separate stream** (or be drawn strictly after every
  physics-affecting draw), so that changing camera parameters leaves dynamics
  bit-identical. This is the same discipline `collect_pairs` relies on for
  frame-0-identical twins ([collect_pairs.py](../../src/data_gen/collect_pairs.py) docstring).
  Verify by asserting `state.json` world-frame reconstruction is unchanged across
  two runs that differ only in camera path.
- **Egomotion twins** are then just: same seed, `--camera_path` A vs B.
  Preferred over kinematic replay — re-simulating the seed is already proven
  deterministic, and replay would require reconstructing world poses from
  camera-frame records.

## 5. Label-contract delta

### 5.1 Per-frame extrinsics move into `state.json`
Today `R_camera_world` is a single matrix in `meta.json`
([clip_capture.py:381](../../src/data_gen/clip_capture.py)) and
`simulate_clip` takes one constant `cam_extrinsics` tuple per camera
([collect_clip.py:282](../../src/data_gen/collect_clip.py)). Under a moving camera
these become per-frame. `log_state_frame` already takes them as arguments — it needs
the frame-`t` values, not the clip-constant ones.

Per frame, additionally logged (camera-frame invariant preserved):
```
R_camera_world(t), camera_pos_world(t), q_camera_world(t)   # extrinsics = raw truth
v_cam(t), omega_cam(t)      # egomotion IN THE INSTANTANEOUS CAMERA FRAME
```
`g_cam(t) = R_camera_world(t) @ (0,0,-1)` is **derived, not stored** (free from the
extrinsics), as is any relative-pose target.

**Egomotion frame — the gauge rule (decided 2026-07-27).** Absolute camera pose is
**not a label**: world origin (3 DoF) and yaw about gravity (1 DoF) are gauge —
unobservable from pixels, so supervising them asks the model to guess a convention.
Only relative motion is well-posed.
- **Store** per-frame world extrinsics (the transform itself — everything derivable).
- **Log** `v_cam`, `omega_cam` in the instantaneous camera frame: identical type and
  frame convention to the per-body `lin_vel_camera` / `ang_vel_camera` records, so
  *the camera is just another body* and existing target builders apply unchanged.
  The analogy has exactly one break: camera *position* in its own frame is
  identically zero, so position needs an anchor (below) while velocity does not.
- **Derive** the training target as **window-anchored relative pose** — `R_rel`,
  `p_rel` w.r.t. the camera at the window's first frame — never frame-to-frame
  chaining (tiny, noise-dominated, drift-accumulating). Window anchoring also
  matches the GATE-B window-alignment lesson (co-limiter #2, +0.29). The anchor is
  a *convention*, not a gauge violation: internal to the clip and fully recoverable.
- **Scale:** `R_rel` is fully observable; `p_rel` only up to scale in monocular.
  Supervise unit direction and log-magnitude **separately** at first, so a scale
  failure is legible instead of corrupting direction learning.
`meta.json` keeps the frame-0 values for backward compatibility **plus** a
`camera_path` provenance block (family, params, seed) — and a
`schema_version` bump (`rigid-probe/v1-dyncam`) so loaders can branch.

### 5.2 What changes meaning (read this before reusing targets)
With a moving camera, `lin_vel_camera` = object motion **mixed with egomotion**.
That is the intended physics (disentangling them is the capability), but it means
**Gate-B targets are not comparable across static and dynamic slices**. Logging
egomotion (§5.1) keeps world-frame quantities recoverable, so targets can be
decomposed rather than conflated.

### 5.3 Corpus hygiene
Dynamic-camera clips form their own slice with its own acceptance run. Probe
configs must declare which slice they consume; group-aware folds unchanged.

## 6. Code hook points

| Hook | Where | Change |
|---|---|---|
| Roll parameter | `world_to_camera_frame` [clip_capture.py:208](../../src/data_gen/clip_capture.py) | add `roll_rad=0.0` (default = today's zero-roll basis); rotate `(right, up)` about `forward`. **Without this there is no roll, moving camera or not.** |
| Path sampling | new `src/data_gen/camera_paths.py` | families of §3; `t=0` pose must equal today's `sample_camera_pose` for `static` |
| Per-step camera pose | `make_annotators` / `make_rig_annotators` [clip_capture.py:34/181](../../src/data_gen/clip_capture.py) | retain the camera prim path; set pose each step before `rep.orchestrator.step()` |
| Per-frame extrinsics | `simulate_clip` [collect_clip.py:282](../../src/data_gen/collect_clip.py) | `cam_extrinsics` becomes a per-camera **path object**, evaluated per `frame_idx` |
| Metadata | `build_metadata` [clip_capture.py:354](../../src/data_gen/clip_capture.py) | schema bump + `camera_path` block |
| Visibility gating | `gate_camera_rig` / `probe_body_coverage` [collect_clip.py:257](../../src/data_gen/collect_clip.py) | gate on **K sampled poses along the path** (e.g. t ∈ {0, ¼, ½, ¾, 1}), not one pose; drop the camera if any sample fails |

Sketch (additive; existing signatures keep working):
```python
@dataclass
class CameraPath:
    kind: str                      # "static" | "orbit" | ... | composed
    params: dict                   # family params (logged verbatim into meta)
    def pose(self, t: int):        # -> (pos_world, look_at_world, roll_rad)
        ...
    def extrinsics(self, t: int):  # -> (R_cw, cam_pos_w, q_cw) via world_to_camera_frame
        ...
```

## 7. Verify-gate extension (GATE A discipline)

[verify_physics_labels.py:440](../../src/data_gen/verify_physics_labels.py) solves
extrinsics **once** from the config camera. Under a path it must:
- verify closed-form checks against **frame-`t`** extrinsics (the ω sign/scale and
  quaternion-composition cases stay valid — they are per-frame identities);
- add a **camera-only self-check**: with bodies static, every logged
  `com_camera(t)` must equal the analytic reprojection of a fixed world point
  through `R_cw(t)` — isolates path/logging bugs from physics;
- add the **`g_cam` consistency identity**: `g_cam(t+1) == R_rel(t) @ g_cam(t)` with
  `R_rel(t) = R_cw(t+1) @ R_cw(t).T`, to machine precision.

**Labels must pass before any GPU/scale spend** — unchanged rule.

## 8. Pilot before scale

Same shape as the W6 pilots: ~20 rollouts × {static, orbit+roll, gait}, then
(a) verify-gate, (b) visual check that the action stays framed, (c) the §9
acceptance numbers. Scale only what passes.

## 9. Acceptance criteria

1. **Verify-gate passes** with per-frame extrinsics (§7), including the two new checks.
2. **Determinism:** world-frame state reconstruction bit-identical between two runs
   differing only in camera path (§4).
3. **Tilt non-degeneracy:** across the slice, `|g_cam,x|` has meaningful spread
   (roll actually present) and `g_cam` is not predictable from a single frame's
   horizon alone — the honest test is a **frozen-probe control**: single-frame tilt
   decodability should be markedly below full-clip decodability, else the shortcut
   survives and floor-cue suppression is needed too.
4. **Not egomotion-dominated:** flow ratio bound of §3 satisfied on ≥ ~80% of frames.
5. **Framing survival:** camera-drop rate from path gating (§6) stays acceptable
   (quality > count — the existing rig philosophy).

## 10. Open knobs (decide before the scale batch)

- **Motion blur.** A moving camera with none is itself a sim cue; RTX blur is
  expensive. Default off for v1, flagged as a domain-realism gap.
- **Floor-cue suppression.** Independent of motion, still the load-bearing
  anti-shortcut control for tilt (a horizon detector that must *track* is harder,
  but not impossible). Candidates: framing that crops the horizon out, textureless
  or absent ground in a sub-slice.
- **Mixing ratio** static : dynamic in the LoRA training slice.
- **fps interaction** — parked (see the fixed-fps decision); dynamic camera makes
  the Δt question live again for egomotion magnitudes.

## 11. Deferred: robot-mounted cameras (H1/G1)

Uniquely buys *caused* egomotion correlated with an **action stream** — the
V-JEPA-2-AC substrate. Deferred because: a locomotion policy in the loop is
contact-rich and seed-fragile (**determinism is the twin machinery's foundation**);
the robot body can collide with / occlude the oracle-tuned scenario bodies; gait-band
egomotion at walking height gives poor tilt coverage; and the per-clip sim bill is
much heavier. **Middle path if egomotion realism proves insufficient:** a
*kinematically animated* robot (canned walk cycle, cameras attached, no policy) —
deterministic and cheap, keeps limbs-in-frame realism; must be pathed clear of
scenario bodies since a kinematic actor shoves dynamic ones.
Promotion condition: the project enters action-conditioned territory.

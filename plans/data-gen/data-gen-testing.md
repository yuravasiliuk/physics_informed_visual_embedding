# How to conduct data-gen tests

Operational guide for validating the Isaac Sim data-collection pipeline
([src/data_gen/](../../src/data_gen/)). The *why* behind the labels and scenes
lives in [probe-dataset.md](probe-dataset.md), [scene-diversification.md](scene-diversification.md),
[physics-diversification.md](physics-diversification.md), and
[camera-framing-fix.md](camera-framing-fix.md); this doc is the **how-to-check-it**.

There are four layers, cheapest first. Run them in order — a failure at a lower
layer makes the ones above meaningless.

| Layer | What it proves | Needs Isaac? | Runtime |
|---|---|---|---|
| 0. Static | syntax, names, placement feasibility | no | seconds |
| 1. Closed-form label checks | the physics labels are numerically correct | yes | ~1–2 min |
| 2. Smoke generation | a real run produces well-formed clips | yes | a few min |
| 3. Visibility / quality | the action is actually visible & trainable | no (post-hoc) | seconds |

---

## 0. Static checks (no Isaac)

Run from [src/data_gen/](../../src/data_gen/). These catch the bulk of edit
mistakes before paying the ~30–60 s Isaac boot.

```bash
cd src/data_gen
# Syntax + (for config.py, which has no omni imports) import-ability:
python3 -m py_compile *.py
python3 -m pyflakes config.py scenarios.py clip_capture.py collect_clip.py
# config has no omni deps — it imports and instantiates standalone:
python3 -c "from config import ClipConfig; ClipConfig()"
```

**Placement feasibility (off-Isaac).** Body placement is pure geometry, so
rejection-sampling can be checked without Isaac. When you change `scale_range`,
`scale_range_env`, `BODY_SCALE_REFERENCE`, or any scenario spawn footprint,
replicate `body_factory.sample_non_overlapping_positions` + `bounding_radius`
over a few hundred seeds at both base and env scale and confirm **0 placement
failures** (the method that caught the env-scale `throw` crash is recorded in
[camera-framing-fix.md](camera-framing-fix.md) Stage 3). This is the guard
against the `RuntimeError: could not place N bodies without overlap` class.

---

## 0b. Dynamic-camera + contact rebuild — what to test, in order (2026-07-27)

*The rebuild adds two things that can corrupt labels **silently**: per-frame
extrinsics (every camera-frame label now depends on the frame-`t` camera, so one
off-by-one silently rewrites the whole corpus) and PhysX contact events. Spec:
[dynamic-camera.md](dynamic-camera.md).*

**The governing rule: isolate the two motions before combining them.** With both
the camera and the bodies moving you cannot tell which side a label error came
from. So test in this order:

| # | Test | Isaac? | What it isolates |
|---|---|---|---|
| 1 | `verify_camera_paths.py` (18 checks) + `verify_camera_motion.py [--all]` | no | trajectory math alone: legacy-basis equality, roll breaking `g.x≡0`, orbit/dolly = zero tilt, USD Euler round trip, SO(3) drift, command round trip, tracking-preserves-roll, determinism, clamps |
| 2 | **Static bodies + MOVING camera** | yes | *pure logging/extrinsics*. Every logged `com_camera(t)` must equal the analytic reprojection of a FIXED world point through `R_cw(t)`. No physics involved → any error is wiring |
| 3 | **Moving bodies + STATIC camera** | yes | the existing closed-form suite (ballistic, `spin_cuboid`, `two_body`) — must still pass **bit-identically**, proving the refactor didn't disturb the zero-coefficient path |
| 4 | **Both moving** | yes | only now: egomotion identity — logged `v_cam`/`ω_cam` must match finite differences of the logged extrinsics; and `g_cam(t+1) == R_rel(t)·g_cam(t)` on LOGGED data (the offline version already passes, so a failure here is wiring, not math) |
| 5 | **Contact impulse** — generate with `--log_contacts`, then `python3 src/data_gen/verify_contacts.py --data <corpus>` | gen only | `J_net ≈ m·Δv − m·g·Δt` per body per frame. **The trust gate for every contact label.** Residuals are split ground vs body-body because a flipped actor0/actor1 reaction fails body-body while ground still passes |
| 6 | Smoke batch per camera family (§2 below) | yes | drop rate from path-aware visibility gating; framing survives the whole path |
| 7 | `material_oracle --event_features` on the pilot | no | **acceptance**: restitution must reproduce ≈ +0.2 R² (event-anchored). A drop means the rebuild changed impact statistics |
| 8 | Tilt non-degeneracy | no | `|g_cam.x|` spread > 0 **and** single-frame decodability ≪ full-clip (else it is still a horizon detector) |
| 9 | Egomotion flow bound | no | egomotion-induced flow must not dominate object flow ([dynamic-camera.md §9.4](dynamic-camera.md)) |

**Dead check to remove:** `offset_com` in `verify_physics_labels` tested the
authored-CoM path, which is abandoned (EXPERIMENTS.md Graveyard). Deleting it is
part of the CoM scrub, not a regression.

---

## 1. Closed-form label-correctness checks (Isaac)

These shoot known trajectories and compare the saved labels against analytic
ground truth. They share `collect_clip`'s capture / camera-frame / save code, so
a bug in pose logging, camera-frame conversion, velocity projection, or save
layout surfaces here. **Run these whenever you touch label or capture code.**

```bash
# Ballistic free-flight: a sphere is impulsed at 45° and its saved camera-frame
# CoM is inverted to world frame and compared to x0 + v0 t + ½ g t². Three
# impulse magnitudes check proportional scaling.
${ISAAC_SIM}/python.sh src/data_gen/verify_collect_clip.py \
    --result ~/data/verify_collect_clip

# Four closed-form cases on the PRODUCTION camera/code path:
#   spin_cuboid — constant torque-free ω, +Z-only orientation increment
#   offset_com  — authored offset CoM equals read_local_com (the W2 scale-fix
#                 regression); world-CoM z second-difference = -g·render_dt²
#   two_body    — per-body state stays matched to the right body (no index swap)
${ISAAC_SIM}/python.sh src/data_gen/verify_physics_labels.py \
    --result ~/data/verify_physics_labels
# one case only:
${ISAAC_SIM}/python.sh src/data_gen/verify_physics_labels.py --case offset_com
```

`${ISAAC_SIM}` is your Isaac Sim install; the scripts pin the asset root to
`file://~/isaac_assets` internally (`collect_clip.ISAAC_ASSETS_DIR`, expanded at
import — see the carb override at the top of that module).

**What passing looks like.** `verify_physics_labels.py` prints one line per case
and a final `summary: N/N cases passed` — you want **all** cases passed. Each run
also writes `verify.json` (the per-check residuals) into every case's output dir,
so you can inspect *how* close, not just pass/fail. `verify_collect_clip.py`
prints per-magnitude residuals; eyeball that the airborne-phase error is small
and scales with impulse.

---

## 2. Smoke generation (Isaac)

Generate a small batch through the production entry point
[collect_clip.py](../../src/data_gen/collect_clip.py) and confirm the clips are
well-formed.

```bash
# Mixed scenarios, 6 physics rollouts (each fans out to ≤ n_cameras clips):
${ISAAC_SIM}/python.sh src/data_gen/collect_clip.py \
    --result ~/data/smoke --n_clips 6 --seed 0

# Exercise one archetype at a time (useful when debugging a scenario):
${ISAAC_SIM}/python.sh src/data_gen/collect_clip.py \
    --result ~/data/smoke_throw --scenario throw --n_clips 6 --seed 0

# Force a specific backdrop (or "none" for procedural ground):
${ISAAC_SIM}/python.sh src/data_gen/collect_clip.py \
    --result ~/data/smoke_wh --environment warehouse_shelves --n_clips 4
```

CLI flags: `--result` (output dir), `--n_clips` (physics rollouts), `--seed`,
`--scenario` (one of `scenarios.BUILDERS`), `--environment` (one of
`environments.library.names` + `none`).

**Note the rollout vs. clip distinction.** One rollout is rendered from up to
`config.n_cameras` viewpoints, each written as an independent `clip_NNNNN/` dir.
Cameras that the visibility gate can't make see the bodies are **dropped**, so a
rollout legitimately yields *fewer* than `n_cameras` clips — even zero for a
hard-to-view scene (you'll see `-> 0 clips (no viewable camera)`). Fewer clean
clips beats more noisy ones; do not "fix" this by loosening the gate.

**What each `clip_NNNNN/` should contain:** `rgb.mp4`, `depth.npz`,
`normals.npz`, `seg.npz`, `seg_id_to_class.json`, `state.json`, `meta.json`.

**Check `meta.json`:**
- `validation.ok == true` (and read `validation.reason` on any `false`).
- `multiview.{rollout_seed, camera_index, n_cameras}` is present and sibling
  clips from one rollout share `rollout_seed`.
- `R_camera_world` is a sane rotation; `camera_position_world` / `look_at` differ
  across a rollout's cameras (distinct viewpoints).
- In an env clip, `scene.environment.name` is set and
  `scene.scenario_specs.scene_scale` ≈ 3.0 (bodies + scenery scaled up together);
  in a procedural clip there's no `environment` and `scene_scale` ≈ 1.0.

---

## 3. Visibility / quality acceptance (post-hoc, no Isaac)

A clip can be well-formed and still be training garbage if the action is
occluded, tiny, or off-frame. This is the acceptance check for any change to
camera placement, gating, or scene scale (see
[camera-framing-fix.md](camera-framing-fix.md) §1, §8). It reads the saved
`seg.npz` + `seg_id_to_class.json` and measures how much of the frame the dynamic
bodies (`body_NN` classes) actually cover.

```bash
python3 - "$HOME/data/smoke" <<'PY'
import sys, json, glob, os
import numpy as np
root = sys.argv[1]
rows = []
for cd in sorted(glob.glob(os.path.join(root, "clip_*"))):
    seg = json.load(open(f"{cd}/seg_id_to_class.json"))["idToLabels"]
    body_ids = [int(k) for k, v in seg.items() if v["class"].startswith("body_")]
    cam = json.load(open(f"{cd}/meta.json"))["multiview"]["camera_index"]
    if not body_ids:
        rows.append((os.path.basename(cd), cam, 0, 0.0)); continue
    arr = np.load(f"{cd}/seg.npz")["seg"]            # (T,H,W)
    cov = max(np.isin(fr, body_ids).mean() for fr in arr) * 100
    rows.append((os.path.basename(cd), cam, len(body_ids), cov))
print(f"{'clip':12}{'cam':>4}{'bodies':>8}{'peak_cov%':>11}")
for name, cam, nb, cov in rows:
    flag = "  <-- INVISIBLE" if nb == 0 else ("  <-- low" if cov < 1.0 else "")
    print(f"{name:12}{cam:>4}{nb:>8}{cov:>11.2f}{flag}")
zero = sum(1 for *_, nb, _ in [(0,0,r[2],0) for r in rows] if nb == 0)
print(f"\n{zero}/{len(rows)} clips have ZERO visible bodies")
PY
```

(For a black-frame scan, `ffprobe -f lavfi movie=clip_*/rgb.mp4,signalstats`
and read `lavfi.signalstats.YAVG`; a clip averaging YAVG ≈ 16 over all frames is
effectively black.)

**Acceptance targets** (from camera-framing-fix.md §8):
- **0** clips with zero visible bodies.
- **0** black clips.
- median peak body coverage **≥ ~10%** (the pre-fix baseline was ~1.2%).
- every surviving camera ≥ `config.min_body_coverage`; the full body set shows up
  across the rig.

---

## Known failure modes (and where they come from)

| Symptom | Likely cause | Where to look |
|---|---|---|
| `RuntimeError: could not place N bodies without overlap` | a spawn footprint too small for the body size (e.g. env scale-up without scaling the scenery) | `scenarios._scene_scale` / `_scaled`; rerun the Layer-0 placement check |
| Clips render with **no environment backdrop** (procedural ground only) | `config.use_isaac_envs = False`, or the env failed `omni.client.stat` and was dropped at discovery | [config.py](../../src/data_gen/config.py) `use_isaac_envs`; `environments.discover` warnings |
| A specific env renders **textureless / white floor** | bad MDL/material binding on the local asset mirror (this is why `office.usd` is excluded) | `environments.CANDIDATE_ENVIRONMENTS` |
| Rollout emits **0 clips** | visibility gate dropped every camera (genuinely un-viewable scene) | expected occasionally; investigate only if frequent |
| Action **occluded / tiny / off-frame** | camera framing or gating regression | run Layer 3; check `frame_fill_frac`, `min_body_coverage`, the pre-roll |
| `validation.ok == false`, `reason=low_body_coverage` | bodies below the coverage floor in the saved clip | same as above; it's the write-time net behind the gate |

---

## When to run what

- **Touched label / capture / camera-frame code** → Layer 0 + Layer 1 (closed-form), always.
- **Touched scenarios / scene scale / placement** → Layer 0 placement check + a Layer 2 smoke run of the affected scenario.
- **Touched camera placement / gating / framing** → Layer 2 + Layer 3 acceptance.
- **Touched env / asset loading** → Layer 2 with `--environment`, then Layer 3 to confirm assets render.
- **Before committing a corpus run** → all four.

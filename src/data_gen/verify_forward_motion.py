"""Forward-motion convention gate — the ONE camera check that requires Isaac.

Captures a single deterministic clip of **steady forward motion** and then proves,
from the rendered pixels, that forward is forward.

**Why this cannot be done off-Isaac.** `verify_camera_motion.py` already proves
`dolly_in` closes on the subject and that `usd_rotation_xyz_deg` round-trips to
`R.T`. Both are statements about *our own* arithmetic being self-consistent. What
no off-Isaac check can reach is whether **USD's** interpretation of that
`rotateXYZ` + translate matches ours. If that mapping were inverted, every one of
the 40 offline checks would still pass and the render would dolly *backwards* —
silently, on every dynamic clip in the corpus. This script closes that hole by
measuring **rendered depth**: move the camera forward `d` metres and the scene in
front of it must get `~d` metres closer. Sign is the convention test.

This is step 2 of the test ladder (data-gen-testing.md §0b), "static bodies +
MOVING camera": the `settle` scenario parks the bodies within ~1-2 s, so for the
rest of the clip every pixel of image motion is the camera's doing and a label
error cannot be blamed on physics.

**Deterministic by construction, not just by seed.** The motion is a hand-written
one-term `MotionModel` (constant surge, no ramp, no roll, no tracking, no rng) —
`sample_motion` is not consulted at all, so this script is unaffected by which
families are implemented and stays a fixed reference as they land.

    # generate + check (needs Isaac; ~300 frames of raytracing, several minutes)
    ${ISAAC_SIM}/python.sh src/data_gen/verify_forward_motion.py \\
        --result ~/data/fwd_check

    # re-run the checks on an already-written clip — pure numpy, NO Isaac
    python3 src/data_gen/verify_forward_motion.py --check_only ~/data/fwd_check/clip_00000
"""

import os
import sys

# argparse has not run yet, but SimulationApp must be constructed before ANY
# omni/isaac import (Kit segfaults otherwise) — so the --check_only decision has
# to be read straight off argv. --check_only is the pure-numpy path over an
# existing clip and must stay runnable on a machine without Isaac at all.
CHECK_ONLY = "--check_only" in sys.argv

if __name__ == "__main__" and not CHECK_ONLY:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    # NOTE: no carb asset-root override here. Importing `collect_clip` applies it
    # at module scope (see the comment block at the top of that file); doing it
    # twice would just be a second copy of a load-bearing path to keep in sync.

import argparse
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_motion import FreeCameraPath, MotionModel   # noqa: E402  pure numpy
from camera_paths import look_at_basis                  # noqa: E402
from depth_map_factory import DepthMapFactory           # noqa: E402

# 10 s at the 30 fps render rate. Long enough that the travel dominates any
# framing ambiguity; short enough to raytrace in one sitting.
DEFAULT_DURATION_S = 10.0
# 0.2 m/s x 10 s = 2.0 m. The rig frames the body cloud at ~2.4-2.9 m, so this
# closes ~70% of the standoff (subject grows ~3x) WITHOUT flying through the
# bodies — an overshoot would make central depth turn around mid-clip and muddy
# the very signal being measured.
DEFAULT_SPEED_MPS = 0.2
# Storage-room backdrop: aisles and shelves give strong along-axis parallax, so
# forward motion is unmistakable both to the eye and to the depth statistic.
# (There is no environment literally named "storage" — see
# environments.CANDIDATE_ENVIRONMENTS; `simple_room` is the smaller alternative.)
DEFAULT_ENVIRONMENT = "warehouse_shelves"
# Central half of the frame: where the rig puts the body cloud, and far from the
# borders that a forward move sweeps new geometry through.
PATCH_FRAC = 0.5

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def steady_forward_motion(speed, dt):
    """Constant surge, nothing else — the definition of "steady forward".

    One `const` term, written by hand rather than drawn from `sample_motion`:
    no ramp (which would make the frame-0 velocity a special case), no roll
    (which would rotate the frame and make "did depth fall?" ambiguous), no rng.
    `surge_sway_heave` inside MotionModel.velocity turns +surge into camera-axis
    -Z, so this is the object under test as much as it is the test fixture.
    """
    return MotionModel(kind="forward_steady", dt=float(dt), terms=(
        {"kind": "const", "v": (float(speed), 0.0, 0.0), "w": (0.0, 0.0, 0.0)},
    ))


def forward_rig_class(speed, base):
    """A `CameraRig` subclass whose every camera flies steadily forward.

    Overrides ONE hook — `_sample_path` — so pose proposal, the t=0 coverage
    gate, the along-path gate, annotator creation and per-frame logging all stay
    exactly the production path. Only the motion is substituted, which is the
    whole point of the fixture, and no production code carries a special case
    for it.

    (Before CameraRig existed this monkeypatched the free function
    `collect_clip.build_camera_paths`. When that function was absorbed into the
    rig the patch silently became a no-op — it set an attribute nothing read —
    and this script would have quietly measured a DEFAULT camera instead of a
    forward-moving one, reporting a pass. A class seam fails loudly instead:
    override a method that does not exist and you get an override that is never
    called, which the `--check_only` trajectory checks would catch immediately.)
    """
    class _ForwardRig(base):
        @staticmethod
        def _sample_path(rng, config, pos, look):
            # roll_rad=0 => the legacy look-at basis, bit-identical to the static
            # rig (gated in verify_camera_paths.py). Frame 0 is therefore exactly
            # the pose the visibility gate approved.
            R_cw0, _ = look_at_basis(pos, look, roll_rad=0.0)
            return FreeCameraPath(steady_forward_motion(speed, config.render_dt),
                                  pos, R_cw0, config.frame_count,
                                  track_target=None, track_tau=0.0)
    return _ForwardRig


def generate(args):
    """Boot the pipeline for ONE rollout and return the written clip dir."""
    import dataclasses

    import collect_clip as cc          # imports carb -> needs SimulationApp alive
    from config import ClipConfig

    base = ClipConfig()
    n_frames = int(round(args.duration / base.render_dt))
    config = dataclasses.replace(
        base,
        scenarios=(args.scenario,),
        n_cameras=1,                   # one camera = one clip = one thing to look at
        frame_count=n_frames,
        # The along-path visibility gate is OFF. It exists to drop cameras that
        # drift off the action, but here a dropped camera means NO footage and
        # therefore no answer — and the question ("is forward forward?") is worth
        # answering even from a badly framed clip.
        camera_path_gate_samples=1,
    )
    cc.CameraRig = forward_rig_class(args.speed, cc.CameraRig)

    print(f"scenario={args.scenario} env={args.environment} seed={args.seed}\n"
          f"{n_frames} frames @ {1/base.render_dt:.0f} fps = {args.duration:.1f} s, "
          f"{args.speed} m/s forward -> {args.speed * (n_frames - 1) * base.render_dt:.2f} m of travel",
          flush=True)
    # Cost warning, because the next output is the finished clip. `simulate_clip`
    # has NO per-frame progress print and `save_clip` creates the output dir as
    # its LAST step — so a long silence with no directory is the expected shape
    # of a healthy run, not a hang. At the production 1024x1024 raytraced
    # settings a heavy backdrop runs on the order of seconds per frame.
    #
    # ★ If you only want the convention answer, cost scales with FRAME COUNT
    #   while the test depends on DISTANCE TRAVELLED — so trade one for the
    #   other: `--duration 2 --speed 1.0` covers the same ~2 m in 60 frames.
    print(f"NOTE: rendering {n_frames} frames "
          f"({n_frames / 64.0:.1f}x a default 64-frame clip) at "
          f"{base.resolution[0]}x{base.resolution[1]}. No progress is printed "
          f"until the clip is written; the output dir appears only at the end.",
          flush=True)
    try:
        results = cc.collect_clip(os.path.abspath(os.path.expanduser(args.result)),
                                  0, config, args.seed, env_override=args.environment)
    except ValueError as e:
        raise SystemExit(f"environment {args.environment!r} unavailable: {e}")
    if not results:
        raise SystemExit("no viewable camera at t=0 — the STATIC rig gate rejected "
                         "the pose, so this is a framing problem, not a motion one")
    out_dir, ok, reason = results[0]
    print(f"wrote {out_dir}  (validate_clip ok={ok} reason={reason})", flush=True)
    return out_dir


def check_clip(clip_dir, speed, duration):
    """Post-hoc gate. Pure numpy — runs with or without Isaac."""
    with open(os.path.join(clip_dir, "camera.json")) as f:
        cam = json.load(f)
    with open(os.path.join(clip_dir, "meta.json")) as f:
        meta = json.load(f)

    R = np.array([r["R_camera_world"] for r in cam])          # (T,3,3) world->camera
    p = np.array([r["camera_pos_world"] for r in cam])         # (T,3)
    v = np.array([r["lin_vel_camera"] for r in cam])           # (T,3) camera axes
    g = np.array([r["g_camera"] for r in cam])
    dt = 1.0 / float(meta.get("render_fps", 30.0))
    T = len(cam)
    travel_expected = speed * (T - 1) * dt

    print(f"\n--- trajectory ({T} frames, {T * dt:.2f} s) ---")
    check("frame count matches the requested duration",
          abs(T * dt - duration) < 2 * dt, f"{T} frames = {T * dt:.2f} s")

    # 1. Steady = no commanded rotation. If orientation drifted, a depth change
    #    could come from looking somewhere else rather than from moving.
    dR = max(float(np.abs(R[t] - R[0]).max()) for t in range(T))
    check("orientation is constant (no rotation commanded)", dR < 1e-9,
          f"max |R(t)-R(0)|={dR:.2e}")
    check("g_camera is constant (follows from the above)",
          float(np.abs(g - g[0]).max()) < 1e-9)

    # 2. Every step advances along the camera's OWN forward axis. Forward in
    #    camera axes is -Z, and R is world->camera, so row 2 of R is the world
    #    direction of camera +Z; forward is its negation.
    fwd_world = -R[0][2]
    steps = (p[1:] - p[:-1]) @ fwd_world
    check("every step advances along the camera's forward axis",
          bool((steps > 0).all()),
          f"min step {steps.min() * 1e3:+.3f} mm, max {steps.max() * 1e3:+.3f} mm")

    travel = float((p[-1] - p[0]) @ fwd_world)
    lateral = float(np.linalg.norm((p[-1] - p[0]) - travel * fwd_world))
    check("total travel matches speed x duration",
          abs(travel - travel_expected) < 1e-6,
          f"{travel:.4f} m vs {travel_expected:.4f} m expected")
    check("motion is purely forward (no lateral drift)", lateral < 1e-9,
          f"|lateral|={lateral:.2e} m")

    # 3. The logged egomotion label itself must read as forward. This is what
    #    downstream targets consume, so a sign error here poisons them even if
    #    the render is fine.
    v_want = np.array([0.0, 0.0, -speed])
    dv = float(np.abs(v[:-1] - v_want).max())    # last frame excluded: see note below
    check("logged lin_vel_camera == (0, 0, -speed) i.e. forward is -Z",
          dv < 1e-9, f"v[0]={np.round(v[0], 6).tolist()} max dev {dv:.2e}")
    check("last-frame egomotion is not a clamped zero",
          float(np.linalg.norm(v[-1])) > 0.5 * speed,
          f"|v[-1]|={np.linalg.norm(v[-1]):.4f} m/s")

    # 4. ★ THE RENDERED CHECK — the only part that tests USD's convention rather
    #    than ours. `distance_to_camera` in the central patch must FALL by
    #    roughly the distance travelled. An inverted pose mapping shows up here
    #    as a rise, and nowhere else.
    print(f"\n--- rendered depth ({os.path.basename(clip_dir)}) ---")
    d = DepthMapFactory.decode_meters(clip_dir)               # (T,H,W) metres
    Td, H, W = d.shape
    hy, hx = int(H * PATCH_FRAC / 2), int(W * PATCH_FRAC / 2)
    patch = d[:, H // 2 - hy:H // 2 + hy, W // 2 - hx:W // 2 + hx].reshape(Td, -1)
    med = np.median(patch, axis=1)
    delta = float(med[-1] - med[0])
    frac_falling = float((np.diff(med) < 0).mean())

    stride = max(1, Td // 8)
    print("   frame   travelled   median depth")
    for t in list(range(0, Td, stride)) + [Td - 1]:
        print(f"   {t:5d}   {speed * t * dt:8.3f} m   {med[t]:8.3f} m")

    check("★ central depth DECREASES (forward is forward in the RENDER)",
          delta < 0.0,
          f"{med[0]:.3f} -> {med[-1]:.3f} m ({delta:+.3f} m over {travel:.3f} m travelled)")
    check("depth falls monotonically for most of the clip", frac_falling > 0.9,
          f"{100 * frac_falling:.1f}% of frames")
    # Soft: the patch can straddle a nearer surface as the camera advances, so
    # the drop need not equal the travel. Only a fraction of it is expected.
    if delta < 0 and abs(delta) < 0.3 * travel:
        print(f"WARN  depth fell only {abs(delta):.3f} m over {travel:.3f} m of travel "
              f"— check the central patch isn't dominated by a surface the camera "
              f"moved PAST rather than toward")

    report = {
        "clip_dir": clip_dir, "frames": T, "dt": dt, "speed_mps": speed,
        "travel_m": travel, "travel_expected_m": travel_expected,
        "lateral_m": lateral, "max_R_drift": dR,
        "depth_median_first": float(med[0]), "depth_median_last": float(med[-1]),
        "depth_delta_m": delta, "frac_frames_depth_falling": frac_falling,
        "failures": list(FAILURES),
    }
    with open(os.path.join(clip_dir, "forward_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", default=os.path.expanduser("~/data/fwd_check"),
                   help="output dir; the clip lands in <result>/clip_00000")
    p.add_argument("--check_only", default=None, metavar="CLIP_DIR",
                   help="skip generation and re-run the checks on an existing "
                        "clip dir. Pure numpy — does NOT boot Isaac.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=None,
                   help=f"seconds of steady forward motion (default {DEFAULT_DURATION_S})")
    p.add_argument("--speed", type=float, default=None,
                   help=f"forward speed in m/s (default {DEFAULT_SPEED_MPS} -> 2 m over 10 s)")
    p.add_argument("--scenario", default="settle",
                   help="default `settle`: the bodies come to rest early, so the "
                        "rest of the clip is the step-2 'static bodies + moving "
                        "camera' isolation")
    p.add_argument("--environment", default=DEFAULT_ENVIRONMENT,
                   help="room backdrop (environments.CANDIDATE_ENVIRONMENTS); "
                        "warehouse_shelves = storage aisles, strong along-axis parallax")
    return p.parse_args()


def motion_params_from_meta(clip_dir):
    """(speed, duration) recovered from the clip's own logged camera_path.

    `--check_only` on a clip generated with non-default settings would otherwise
    be checked against the wrong expectation and FAIL for no reason — the clip
    records what it was, so read it rather than trusting the CLI defaults.
    """
    try:
        with open(os.path.join(clip_dir, "meta.json")) as f:
            cp = json.load(f)["camera_path"]
        if cp.get("motion", {}).get("kind") != "forward_steady":
            return None
        return float(cp["motion"]["terms"][0]["v"][0]), cp["n_frames"] * cp["dt"]
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return None


def main():
    args = parse_args()
    if args.check_only:
        clip_dir = args.check_only
        logged = motion_params_from_meta(clip_dir)
        if logged is not None:
            speed = args.speed if args.speed is not None else logged[0]
            duration = args.duration if args.duration is not None else logged[1]
            print(f"read from meta.json: speed={logged[0]} m/s, "
                  f"duration={logged[1]:.2f} s")
        else:
            speed = args.speed if args.speed is not None else DEFAULT_SPEED_MPS
            duration = args.duration if args.duration is not None else DEFAULT_DURATION_S
            print("meta.json has no forward_steady camera_path — falling back to "
                  "CLI/default speed and duration")
    else:
        speed = args.speed if args.speed is not None else DEFAULT_SPEED_MPS
        duration = args.duration if args.duration is not None else DEFAULT_DURATION_S
        args.speed, args.duration = speed, duration
        clip_dir = generate(args)
    check_clip(clip_dir, speed, duration)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print(f"forward motion verified — {clip_dir}/rgb.mp4 should show the scene "
          f"approaching, and forward_check.json has the numbers")
    return 0


if __name__ == "__main__":
    code = main()
    if not CHECK_ONLY:
        simulation_app.close()
    raise SystemExit(code)

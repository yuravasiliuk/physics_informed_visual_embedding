"""Trust gate for logged contact events — runs on a corpus, WITHOUT Isaac.

The closed-form identity every contact must satisfy (impulse-momentum theorem
over one rendered frame, with gravity's contribution removed):

    J_net ≈ m·Δv − m·g·Δt

`J_net` is the summed solver impulse acting on the body in that frame,
`Δv` its world-frame velocity change across the frame, `Δt = 1/render_fps`.
This is the contact analogue of the ballistic/spin closed-form checks in
verify_physics_labels.py, and it must pass before any scale spend (CLAUDE.md
§Verification discipline).

It also catches the classic sign error: PhysX reports one impulse per
(actor0, actor1) pair, and the reaction on actor1 is its negation. If
`net_impulse_per_body` had that backwards, every body-body event would fail
here while ground contacts still passed — which is exactly why the residual is
reported split by contact type.

    python3 src/data_gen/verify_contacts.py --data ~/data/dyncam_smoke
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_log import GROUND_NAME, net_impulse_per_body, restitution_from_impulse  # noqa: E402

G_WORLD = np.array([0.0, 0.0, -9.81])


def _clip_dirs(root):
    out = []
    for dirpath, _, names in os.walk(root):
        if "meta.json" in names and "contacts.json" in names:
            out.append(dirpath)
    return sorted(out)


def check_clip(cd):
    """Per-body, per-frame residual of J ≈ mΔv − mgΔt. Returns a list of rows."""
    with open(os.path.join(cd, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(cd, "state.json")) as f:
        state = json.load(f)
    with open(os.path.join(cd, "contacts.json")) as f:
        contacts = json.load(f)
    if not contacts:
        return []

    R_cw = np.asarray(meta["R_camera_world"], dtype=np.float64)
    dt = 1.0 / float(meta.get("render_fps", 30.0))
    mass_by_name = {b["name"]: float(b.get("mass_measured", b.get("mass", np.nan)))
                    for b in meta["bodies"]}

    # Per-frame world-frame velocity per body. state.json is camera-frame; for a
    # STATIC camera one matrix inverts it. (Dynamic clips carry camera.json with
    # per-frame extrinsics — handled below.)
    per_frame_R = None
    cam_json = os.path.join(cd, "camera.json")
    if os.path.exists(cam_json):
        with open(cam_json) as f:
            per_frame_R = {int(r["frame"]): np.asarray(r["R_camera_world"], float)
                           for r in json.load(f)}

    vel = {}
    for r in state:
        R = per_frame_R.get(int(r["frame"]), R_cw) if per_frame_R else R_cw
        vel.setdefault(r["body"], {})[int(r["frame"])] = R.T @ np.asarray(
            r["lin_vel_camera"], dtype=np.float64)

    by_frame = {}
    for e in contacts:
        by_frame.setdefault(int(e["frame"]), []).append(e)

    rows = []
    for frame, events in by_frame.items():
        names = {e["body_a"] for e in events} | {e["body_b"] for e in events}
        for name in names:
            m = mass_by_name.get(name)
            if m is None or not np.isfinite(m):
                continue                      # ground / scenery: no momentum book
            v = vel.get(name, {})
            if frame not in v or (frame + 1) not in v:
                continue                      # last frame has no Δv
            dv = v[frame + 1] - v[frame]
            expected = m * dv - m * G_WORLD * dt
            measured = net_impulse_per_body(events, name)
            scale = max(np.linalg.norm(expected), 1e-6)
            pair = any(e["body_a"] != GROUND_NAME and e["body_b"] != GROUND_NAME
                       and (e["body_a"] == name or e["body_b"] == name)
                       for e in events)
            rows.append({
                "clip": os.path.basename(cd), "frame": frame, "body": name,
                "rel_residual": float(np.linalg.norm(measured - expected) / scale),
                "kind": "body-body" if pair else "ground",
                "impulse_mag": float(np.linalg.norm(measured)),
                "mass": m,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tol", type=float, default=0.25,
                    help="relative residual counted as a pass. Loose on purpose: "
                         "several PhysX substeps fall inside one rendered frame, "
                         "so Δv is a coarser measure than the impulse itself — "
                         "the check is for SIGN/SCALE errors, not solver noise.")
    args = ap.parse_args()
    root = os.path.expanduser(args.data)

    dirs = _clip_dirs(root)
    if not dirs:
        sys.exit(f"no clips with contacts.json under {root} — was the corpus "
                 "generated with log_contacts enabled?")

    rows = []
    for cd in dirs:
        rows.extend(check_clip(cd))
    if not rows:
        sys.exit("contacts.json present but no checkable events (no body with "
                 "both mass and a Δv). Corpus too short, or contacts empty.")

    res = np.array([r["rel_residual"] for r in rows])
    print(f"{len(dirs)} clips | {len(rows)} checkable body-frame contact events")
    print(f"relative residual |J - (mΔv - mgΔt)| / |mΔv - mgΔt|:")
    print(f"  median {np.median(res):.3f}   p90 {np.percentile(res, 90):.3f}   "
          f"max {res.max():.3f}")
    for kind in ("ground", "body-body"):
        sel = np.array([r["kind"] == kind for r in rows])
        if sel.any():
            print(f"  {kind:10s} n={sel.sum():5d}  median {np.median(res[sel]):.3f}"
                  f"  pass@{args.tol} {np.mean(res[sel] < args.tol):.1%}")
    ok = float(np.mean(res < args.tol))
    print(f"\npass rate @tol={args.tol}: {ok:.1%}")
    if ok < 0.8:
        print("FAIL — impulses do not reconcile with momentum. Check FIRST: the "
              "actor0/actor1 sign convention in contact_log.net_impulse_per_body "
              "(a flipped reaction fails body-body while ground still passes).")
        return 1
    print("PASS — logged impulses reconcile with momentum change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

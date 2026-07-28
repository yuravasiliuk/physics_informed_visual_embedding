"""Scene 02 — Egocentric representation (the failure model, for comparison).

Companion to scene 01. SAME external viewpoint, palette and gizmo language, but
the roles invert:
  * The CAMERA frame is fixed at the origin — it *is* the coordinate system now
    (ego triad: x=right/red, y? we keep colours by MEANING — right/red,
    up/green, optical-axis/blue). Up is aligned to screen-up; forward (optical
    axis) points into the scene.
  * The object is STATIC in the world; the ONLY motion is the camera's own
    egomotion — it both PANS (yaw about the up axis) and TRANSLATES (dollies)
    through the world. Re-expressed in the camera frame, the fixed world (grid +
    static object, pinned together by a stem+footprint) sweeps rigidly around the
    camera. This isolates, with a single unambiguous cause, the two structural
    failures from deck slide 3:
        (1) camera motion contaminates every measurement (a motionless object
            traces an arc in the ego frame — the trail is pure camera-motion);
        (2) what leaves the view frustum has NO representation — the object
            sweeps out of the FOV and drops out of the state entirely
            ("left the view → left the state" = the permanence failure).

Paced for the recorded voiceover (reports/video/narration/
scene02_egocentric_representation.wav, 19.8 s): the beats land on the wav's
pause boundaries (motion ~4.8 s, exit verdict ~10.4 s, closing card ~12.9 s),
and after the object exits the camera KEEPS DRIVING along the grid — through
the closing card — rather than freezing the last frame.

Semantic axes are mapped onto manim world coords so screen-up = up:
    up = +Z (green) · forward/optical = +Y (blue) · right = +X (red).
Egomotion is a yaw about the camera's up axis (rot_z).

    python scripts/video_scenes/scene02_egocentric_representation.py           # quick preview
    python scripts/video_scenes/scene02_egocentric_representation.py -q h      # 1080p
"""
from __future__ import annotations

import argparse
import numpy as np
from manim import (
    Circle, Create, Dot3D, FadeIn, FadeOut, Line, Text, VGroup,
    ThreeDScene, ValueTracker, config, tempconfig, linear,
    ORIGIN, DEGREES, UP, DOWN, LEFT, RIGHT, UL,
)

# Reuse palette + helpers from scene 01 (no duplication; scene 01 renders only
# under its own __main__, so importing it is side-effect free).
from scene01_allocentric_worldlines import (
    BG, TEXT, MUTED, GRID, CYAN, GREEN, YELLOW, RED, BLUE,
    polyline, _swatch, _billboard,
)

# --- ego-frame geometry (semantic up=+Z, forward=+Y, right=+X) ---------------
Y0 = -1.3           # ground plane height (below the camera)
ALPHA = 24 * np.pi / 180.0   # FOV half-angle (horizontal)
ASPECT = 0.6        # vertical/horizontal FOV ratio
FAR = 6.5           # FOV depth
TAN = np.tan(ALPHA)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _fit(mob, frac: float = 0.92):
    """Scale a fixed-in-frame mobject down so it never overflows the frame width."""
    if mob.width > config.frame_width * frac:
        mob.scale(config.frame_width * frac / mob.width)
    return mob


# The object never moves in the world; the ONLY motion is the camera's.
OBJ_WORLD = np.array([1.2, 3.0, 0.3])       # static world position (to the right)
OBJ_FOOT = np.array([1.2, 3.0, -1.3])       # its footprint on the ground grid (z=Y0)


def psi(s: float) -> float:
    """Camera pan = yaw about the up axis, ~46 deg over the clip."""
    return np.deg2rad(46.0 * s)


CAM_VEL = np.array([1.3, 1.4, 0.0])   # camera world velocity (per unit s): sideways + forward


def cam_shift(s: float) -> np.ndarray:
    """Camera SPATIAL translation through the world (dolly): sideways + forward."""
    return CAM_VEL * s


def cam_ground_world(u: float) -> np.ndarray:
    """The camera's own footprint on the ground, in the WORLD frame, at time u."""
    return CAM_VEL * u + np.array([0.0, 0.0, Y0])


def ego_transform(s: float):
    """World frame -> camera(ego) frame for the full rigid egomotion (yaw + dolly):
        p_ego = R(psi) @ (p_world - cam_shift).
    Applied identically to the grid AND the object, so their relative geometry is
    preserved exactly (the object stays rigidly fixed in the world); only the
    camera moves. A static world point therefore sweeps across the fixed view."""
    R = rot_z(psi(s)); c = cam_shift(s)
    return lambda p: R @ (np.asarray(p, dtype=float) - c)


def obj_ego(s: float) -> np.ndarray:
    """Where the *static* object appears in the camera frame at time s."""
    return ego_transform(s)(OBJ_WORLD)


def visible(p: np.ndarray) -> bool:
    """Inside the rectangular view frustum along +Y?"""
    fwd = p[1]
    if fwd <= 0.25:
        return False
    return (abs(p[0]) / fwd < TAN) and (abs(p[2]) / fwd < TAN * ASPECT)


def find_exit(n: int = 400) -> float:
    """First s at which the object, once seen, leaves the FOV."""
    seen = False
    for i in range(n + 1):
        s = i / n
        v = visible(obj_ego(s))
        if v:
            seen = True
        elif seen:
            return s
    return 1.0


def ego_triad() -> VGroup:
    L = 1.15
    axes = VGroup(
        Line(ORIGIN, [L, 0, 0], stroke_color=RED,   stroke_width=6),      # right
        Line(ORIGIN, [0, 0, L], stroke_color=GREEN, stroke_width=6),      # up (+Z)
        Line(ORIGIN, [0, 1.5 * L, 0], stroke_color=BLUE, stroke_width=7),  # optical (+Y)
    )
    tips = VGroup(
        Dot3D([L, 0, 0], radius=0.06, color=RED),
        Dot3D([0, 0, L], radius=0.06, color=GREEN),
        Dot3D([0, 1.5 * L, 0], radius=0.08, color=BLUE),
        Dot3D(ORIGIN, radius=0.1, color=CYAN),
    )
    return VGroup(axes, tips)


def fov_frustum() -> VGroup:
    hx, hz = FAR * TAN, FAR * TAN * ASPECT
    corners = [np.array([sx * hx, FAR, sz * hz]) for sx, sz in
               [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    edges = VGroup(*[Line(ORIGIN, c, stroke_color=CYAN, stroke_width=2.2,
                          stroke_opacity=0.55) for c in corners])
    far = VGroup(*[Line(corners[i], corners[(i + 1) % 4], stroke_color=CYAN,
                        stroke_width=2.2, stroke_opacity=0.55) for i in range(4)])
    return VGroup(edges, far)


def build_object(s: float) -> VGroup:
    """The static object, pinned to the grid so it visibly rides with the world:
    a ball + a vertical stem down to its footprint + the footprint marker on the
    grid. All three are the SAME world points under ego_transform(s), so they move
    rigidly with the grid — making 'the object is static in the world' unmistakable
    (any apparent screen drift is parallax, not motion)."""
    E = ego_transform(s)
    ball_p, foot_p = E(OBJ_WORLD), E(OBJ_FOOT)
    stem = Line(foot_p, ball_p, stroke_color=YELLOW, stroke_width=2.2, stroke_opacity=0.55)
    foot = Dot3D(foot_p, radius=0.07, color=YELLOW)
    ball = Dot3D(ball_p, radius=0.15, color=YELLOW)
    return VGroup(stem, foot, ball)


def world_grid(E) -> VGroup:
    """Ground grid on the base plane z=Y0, transformed into the ego frame by E."""
    lines = []
    xs = np.arange(-5, 5.01, 1.0)
    ys = np.arange(0.5, 8.01, 1.0)
    for y in ys:
        lines.append((np.array([-5, y, Y0]), np.array([5, y, Y0])))
    for x in xs:
        lines.append((np.array([x, 0.5, Y0]), np.array([x, 8.0, Y0])))
    segs = [Line(E(p0), E(p1), stroke_color=GRID, stroke_width=1.5) for p0, p1 in lines]
    g = VGroup(*segs); g.set_stroke(opacity=0.7)
    return g


# --------------------------------- the scene --------------------------------
class Scene02EgocentricRepresentation(ThreeDScene):
    params: dict = {}

    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=68 * DEGREES, theta=-56 * DEGREES, zoom=0.95)
        s_exit = find_exit()

        # --- HUD ---------------------------------------------------------
        title = Text("Egocentric representation", font_size=40, color=TEXT, weight="BOLD")
        title.to_edge(UP).shift(0.1 * DOWN)
        legend = VGroup(
            _swatch(RED,   "x  — right"),
            _swatch(GREEN, "y  — up"),
            _swatch(BLUE,  "z  — optical axis (view direction)"),
            _swatch(CYAN,  "camera: frustum, footprint & ground track"),
            _swatch(YELLOW, "object — static in the world (footprint pinned to grid)"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).scale(0.58)
        legend.to_corner(UL, buff=0.45).shift(0.85 * DOWN)
        caption = _fit(Text("the camera moves and pans through the world — the object "
                            "never moves; only the camera does",
                            font_size=24, color=MUTED)).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(title, legend, caption)
        for m in (title, legend, caption):
            m.set_opacity(0)

        # --- establish the fixed camera frame + FOV ----------------------
        triad = ego_triad()
        fov = fov_frustum()
        self.play(FadeIn(triad, run_time=1.0), title.animate.set_opacity(1.0))
        self.add_fixed_orientation_mobjects(
            _billboard("x", RED,   [1.32, 0, 0], 24),
            _billboard("y (up)", GREEN, [0, 0, 1.32], 22),
            _billboard("z (optical)", BLUE, [0, 1.85, 0], 22),
        )
        # camera's own projection to the ground: plumb-stem + footprint. In the ego
        # frame the camera is always directly above its footprint, so these are
        # screen-fixed while the grid streams past underneath (i.e. the camera moves).
        cam_stem = Line(ORIGIN, [0, 0, Y0], stroke_color=CYAN, stroke_width=2.2,
                        stroke_opacity=0.55)
        cam_foot = Dot3D([0, 0, Y0], radius=0.08, color=CYAN)
        self.play(Create(fov, run_time=1.0), FadeIn(cam_stem, run_time=1.0),
                  FadeIn(cam_foot, run_time=1.0), legend.animate.set_opacity(1.0))

        # --- world appears, already in the camera's frame ----------------
        s = ValueTracker(0.0)
        grid = world_grid(ego_transform(0.0))
        obj = build_object(0.0)
        self.play(FadeIn(grid, run_time=1.0), FadeIn(obj, run_time=0.6),
                  caption.animate.set_opacity(1.0))
        self.wait(1.8)   # hold the static setup while the VO intro finishes (~0–4.6 s)

        # --- dynamics: world swirls; object crosses & then exits the FOV --
        trail = polyline(obj_ego, 0.0, YELLOW, 4)
        # the camera's own ground track: where it has driven, streaming back from
        # its (screen-fixed) footprint — this is what makes the dolly legible.
        cam_track = polyline(lambda u: ego_transform(0.0)(cam_ground_world(u)), 0.0, CYAN, 3)
        self.add(trail, cam_track)

        def upd_grid(m):
            m.become(world_grid(ego_transform(s.get_value())))

        def upd_obj(m):
            sv = s.get_value()
            m.become(build_object(sv))
            m.set_opacity(1.0 if visible(obj_ego(sv)) else 0.0)

        def upd_trail(m):
            m.become(polyline(obj_ego, min(s.get_value(), s_exit), YELLOW, 4))

        def upd_cam_track(m):
            sv = s.get_value(); E = ego_transform(sv)
            m.become(polyline(lambda u: E(cam_ground_world(u)), sv, CYAN, 3))

        grid.add_updater(upd_grid); obj.add_updater(upd_obj)
        trail.add_updater(upd_trail); cam_track.add_updater(upd_cam_track)

        # phase 1: object visible, world already contaminating its trail.
        # RATE = wall-seconds per unit s; 8.0 puts the exit beat at ~10.4 s,
        # right where the VO's "left the view → left the state" chunk starts.
        RATE = 8.0
        self.play(s.animate.set_value(s_exit), run_time=RATE * s_exit, rate_func=linear)

        # the exit beat: object drops out of the state
        for m in (obj, trail):
            m.clear_updaters()
        obj.clear_updaters()
        ghost = Circle(radius=0.16, stroke_color=MUTED, stroke_width=3,
                       fill_opacity=0).move_to(obj_ego(s_exit))
        lost = _fit(Text("a static object left the view   →   left the state",
                         font_size=28, color=YELLOW, weight="BOLD")).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(lost); lost.set_opacity(0)
        # fade the caption out first, then bring the verdict in (no overlap)
        self.play(FadeOut(obj, run_time=0.4), FadeIn(ghost, run_time=0.4),
                  caption.animate.set_opacity(0.0))
        self.play(lost.animate.set_opacity(1.0), run_time=0.4)

        # phase 2: world keeps moving; the trail is frozen, the object is gone.
        # The camera keeps driving along the grid at the SAME world speed for
        # the rest of the recorded narration — through the closing card — so
        # the 19.8 s wav is covered by real motion, not a frozen last frame.
        grid.add_updater(upd_grid)
        seg_a, seg_b, seg_c = 1.4, 0.8, 5.2   # verdict holds · card swap · drive on
        s1 = s_exit + seg_a / RATE
        s2 = s1 + seg_b / RATE
        s3 = s2 + seg_c / RATE                # ≈ 1.62: cam still well inside the grid
        self.play(s.animate.set_value(s1), run_time=seg_a, rate_func=linear)

        # --- closing (camera still moving underneath) ---------------------
        punch = _fit(Text("egocentric frame: the camera is the origin —\n"
                          "its motion contaminates every measurement,\n"
                          "and what it can't see doesn't exist",
                          font_size=26, color=CYAN, weight="BOLD",
                          line_spacing=0.8), frac=0.8).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(punch); punch.set_opacity(0)
        self.play(s.animate.set_value(s2), FadeOut(lost),
                  punch.animate.set_opacity(1.0), run_time=seg_b, rate_func=linear)
        self.play(s.animate.set_value(s3), run_time=seg_c, rate_func=linear)
        grid.clear_updaters(); cam_track.clear_updaters()
        self.wait(0.6)


# --------------------------------- rendering --------------------------------
_QUALITY = {"l": (854, 480, 15), "m": (1280, 720, 30), "h": (1920, 1080, 30)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", choices=list(_QUALITY), default="l")
    ap.add_argument("--out", default="scene02_egocentric_representation")
    args = ap.parse_args()
    w, h, fps = _QUALITY[args.quality]
    with tempconfig({
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "background_color": BG, "output_file": args.out,
        "media_dir": "reports/video/_media", "format": "mp4",
        "disable_caching": True, "verbosity": "WARNING",
    }):
        Scene02EgocentricRepresentation().render()
        print("rendered:", config.output_file, f"({w}x{h}@{fps})")

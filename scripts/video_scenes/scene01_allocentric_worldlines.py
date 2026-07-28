"""Scene 01 — Exocentric world-lines (ego camera frame in a fixed world grid).

(Terminology: "exocentric" replaced "allocentric" 2026-07-14; the module keeps
its original filename because the sibling scenes import the palette from it.)

Deck slide 4 ("The insight"), animated. Message, made visual:
  * A fixed, gravity-pinned WORLD grid = the exocentric stage.
  * A schematic CAMERA symbol carrying its own ego-frame triad
    (x=right/red, y=up/green, z=optical-axis/blue), origin AT the camera.
  * The camera moves + rotates through the grid tracking an object. The ego
    triad travels glued to the camera (egocentric); the grid never moves
    (exocentric). Both the camera and the object leave a persistent trail —
    their *world-lines*.

Camera-orientation convention (answers "how to pose the camera"):
  build R from a look-at, never by hand-set Euler angles.
    z_cam = normalize(target - eye)          # optical axis / view dir  (blue)
    x_cam = normalize(world_up x z_cam)       # right                    (red)
    y_cam = z_cam x x_cam                      # up                       (green)
  R = [x_cam | y_cam | z_cam] (columns, local->world); pose t = eye.
  The frustum opens along +z_cam.

Style follows ~/sandboxes/teacher (manim 0.20.1, presentation palette). This
file is self-contained so it renders from anywhere:
    python scripts/video_scenes/scene01_allocentric_worldlines.py            # quick preview
    python scripts/video_scenes/scene01_allocentric_worldlines.py -q h       # 1080p
    manim -pqh scripts/video_scenes/scene01_allocentric_worldlines.py Scene01ExocentricWorldlines
"""
from __future__ import annotations

import argparse
import numpy as np
from manim import (
    Arrow3D, Create, Dot3D, FadeIn, FadeOut, Line, Sphere, Square, Text, VGroup,
    ThreeDScene, ValueTracker, config, tempconfig, smooth,
    ORIGIN, PI, DEGREES, UP, DOWN, LEFT, RIGHT, DL, UL,
)

# ---- palette (mirrors teacher/src/video/visual/presentation_scenes.py) ----
BG     = "#0B1020"
TEXT   = "#F4F7FF"
MUTED  = "#A9B4CC"
GRID   = "#26304A"
CYAN   = "#42D3FF"   # camera / ego-line
GREEN  = "#66E3A4"   # y-axis (up) / object world-line
MAGENTA = "#FF6FAE"
YELLOW = "#FFD166"
RED    = "#FF6B6B"    # x-axis (right)
BLUE   = "#5B8CFF"    # z-axis (optical axis)

WORLD_UP = np.array([0.0, 0.0, 1.0])   # gravity-pinned up


# ----------------------------- geometry helpers -----------------------------
def look_at(eye: np.ndarray, target: np.ndarray, up=WORLD_UP) -> np.ndarray:
    """Rotation whose columns are the camera's [x_right | y_up | z_optical]."""
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-9)
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-6:                       # looking straight along world-up: pick a fallback
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        n = np.linalg.norm(x)
    x = x / (n + 1e-9)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])  # local -> world


def eye_path(s: float) -> np.ndarray:
    """Camera position sweeps an arc over the grid, dipping in height."""
    th = -0.45 + s * 2.55
    r = 3.6
    return np.array([r * np.cos(th), r * np.sin(th), 1.35 + 0.55 * np.sin(2.0 * th)])


def obj_path(s: float) -> np.ndarray:
    """The object the camera tracks — a curving trajectory near the center."""
    ph = s * 2.0 * PI
    return np.array([1.3 * np.cos(ph + PI), 1.05 * np.sin(1.3 * ph), 0.45 + 0.35 * (1 - np.cos(ph))])


def camera_gizmo(R: np.ndarray, t: np.ndarray) -> VGroup:
    """Frustum + ego triad at pose (R, t). Frustum opens along +z_cam."""
    def w(local):                      # local -> world
        return t + R @ np.asarray(local, dtype=float)

    f, bw, bh = 0.9, 0.62, 0.42        # frustum depth + half-extents of image plane
    corners = [w([sx * bw, sy * bh, f]) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    apex = w([0, 0, 0])

    frustum = VGroup(
        *[Line(apex, c, stroke_color=MUTED, stroke_width=2.5) for c in corners],
        *[Line(corners[i], corners[(i + 1) % 4], stroke_color=MUTED, stroke_width=2.5)
          for i in range(4)],
    )

    L = 0.95
    axes = VGroup(
        Line(t, w([L, 0, 0]), stroke_color=RED,   stroke_width=5),       # x right
        Line(t, w([0, L, 0]), stroke_color=GREEN, stroke_width=5),       # y up
        Line(t, w([0, 0, 1.35 * L]), stroke_color=BLUE, stroke_width=6),  # z optical (longer)
    )
    tips = VGroup(
        Dot3D(w([L, 0, 0]),        radius=0.055, color=RED),
        Dot3D(w([0, L, 0]),        radius=0.055, color=GREEN),
        Dot3D(w([0, 0, 1.35 * L]), radius=0.07,  color=BLUE),
        Dot3D(apex,                radius=0.09,  color=CYAN),   # camera center
    )
    return VGroup(frustum, axes, tips)


def polyline(fn, s: float, color: str, width: float, n: int = 64) -> VGroup:
    """Growing trail: fn sampled on [0, s]. Empty (invisible) until s advances."""
    if s <= 1e-4:
        return VGroup(Line(fn(0.0), fn(0.0) + np.array([0, 0, 1e-3]),
                           stroke_color=color, stroke_width=width, stroke_opacity=0.0))
    us = np.linspace(0.0, s, max(2, int(n * s)))
    pts = [fn(u) for u in us]
    return VGroup(*[Line(pts[i], pts[i + 1], stroke_color=color, stroke_width=width)
                    for i in range(len(pts) - 1)])


# --------------------------------- the scene --------------------------------
class Scene01ExocentricWorldlines(ThreeDScene):
    params: dict = {}

    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=66 * DEGREES, theta=-52 * DEGREES, zoom=0.95)

        # --- HUD (fixed in frame) ---------------------------------------
        title = Text("Exocentric world-lines", font_size=40, color=TEXT, weight="BOLD")
        title.to_edge(UP).shift(0.1 * DOWN)
        legend = VGroup(
            _swatch(RED,   "x  — right"),
            _swatch(GREEN, "y  — up"),
            _swatch(BLUE,  "z  — optical axis (view direction)"),
            _swatch(CYAN,  "camera world-line (ego-line)"),
            _swatch(YELLOW, "object world-line"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).scale(0.58)
        legend.to_corner(UL, buff=0.45).shift(0.85 * DOWN)
        caption = Text("world grid = fixed (exocentric)   ·   ego frame = glued to the camera",
                       font_size=24, color=MUTED)
        caption.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(title, legend, caption)
        title.set_opacity(0); legend.set_opacity(0); caption.set_opacity(0)

        # --- world stage: ground grid + world axes + gravity ------------
        grid = ground_grid(extent=4.0, step=1.0)
        world_axes = VGroup(
            Arrow3D(ORIGIN, [1.6, 0, 0], color=MUTED, thickness=0.012, base_radius=0.045),
            Arrow3D(ORIGIN, [0, 1.6, 0], color=MUTED, thickness=0.012, base_radius=0.045),
            Arrow3D(ORIGIN, [0, 0, 1.6], color=MUTED, thickness=0.012, base_radius=0.045),
        )
        # gravity: muted + thin + cornered, so it reads as a pin, not a world-line
        gravity = Arrow3D([3.4, 3.4, 1.9], [3.4, 3.4, 0.15], color=MUTED,
                          thickness=0.011, base_radius=0.05)

        self.play(FadeIn(grid, run_time=1.2))
        self.play(Create(world_axes, run_time=1.0), FadeIn(gravity, run_time=1.0))
        self.add_fixed_orientation_mobjects(
            _billboard("g", MUTED, [3.4, 3.4, 2.2], 24),
            _billboard("world frame (fixed)", MUTED, [-0.15, -0.15, 1.9], 22),
        )
        self.play(title.animate.set_opacity(1.0), run_time=0.6)
        self.begin_ambient_camera_rotation(rate=0.045)

        # --- introduce the camera symbol + ego triad --------------------
        s = ValueTracker(0.0)
        R0, t0 = look_at(eye_path(0.0), obj_path(0.0)), eye_path(0.0)
        giz = camera_gizmo(R0, t0)
        self.play(FadeIn(giz, run_time=1.0),
                  legend.animate.set_opacity(1.0), caption.animate.set_opacity(1.0))
        self.wait(0.4)

        # object being observed
        obj = Sphere(radius=0.16, resolution=(16, 16)).set_color(YELLOW).move_to(obj_path(0.0))
        self.play(FadeIn(obj, run_time=0.5))

        # --- dynamics: camera + object move, ego frame rides along ------
        cam_trail = polyline(eye_path, 0.0, CYAN, 5)
        obj_trail = polyline(obj_path, 0.0, YELLOW, 4)
        self.add(cam_trail, obj_trail)

        def upd_giz(m):
            eye = eye_path(s.get_value()); R = look_at(eye, obj_path(s.get_value()))
            m.become(camera_gizmo(R, eye))
        def upd_obj(m):
            m.move_to(obj_path(s.get_value()))
        def upd_cam_trail(m):
            m.become(polyline(eye_path, s.get_value(), CYAN, 5))
        def upd_obj_trail(m):
            m.become(polyline(obj_path, s.get_value(), YELLOW, 4))

        giz.add_updater(upd_giz); obj.add_updater(upd_obj)
        cam_trail.add_updater(upd_cam_trail); obj_trail.add_updater(upd_obj_trail)

        self.play(s.animate.set_value(1.0), run_time=7.0, rate_func=smooth)

        for m in (giz, obj, cam_trail, obj_trail):
            m.clear_updaters()
        self.wait(0.6)

        # --- closing beat ------------------------------------------------
        self.stop_ambient_camera_rotation()
        punch = Text("the camera is just another world-line",
                     font_size=30, color=CYAN, weight="BOLD").to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(punch); punch.set_opacity(0)
        self.play(FadeOut(caption), punch.animate.set_opacity(1.0), run_time=0.8)
        self.wait(1.2)


def _swatch(color: str, label: str) -> VGroup:
    chip = Square(side_length=0.22, fill_color=color, fill_opacity=1,
                  stroke_width=0).set_z_index(2)
    txt = Text(label, font_size=26, color=TEXT)
    row = VGroup(chip, txt).arrange(RIGHT, buff=0.22)
    return row


def _billboard(text: str, color: str, pos, size: int) -> Text:
    t = Text(text, font_size=size, color=color)
    t.move_to(pos)
    return t


def ground_grid(extent: float = 4.0, step: float = 1.0) -> VGroup:
    lines = []
    xs = np.arange(-extent, extent + 1e-6, step)
    for x in xs:
        lines.append(Line([x, -extent, 0], [x, extent, 0], stroke_color=GRID, stroke_width=1.6))
        lines.append(Line([-extent, x, 0], [extent, x, 0], stroke_color=GRID, stroke_width=1.6))
    g = VGroup(*lines)
    g.set_stroke(opacity=0.75)
    return g


# --------------------------------- rendering --------------------------------
_QUALITY = {
    "l": (854, 480, 15),
    "m": (1280, 720, 30),
    "h": (1920, 1080, 30),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", choices=list(_QUALITY), default="l")
    ap.add_argument("--out", default="reports/video/scene01_exocentric_worldlines")
    args = ap.parse_args()
    w, h, fps = _QUALITY[args.quality]
    with tempconfig({
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "background_color": BG, "output_file": args.out.split("/")[-1],
        "media_dir": "reports/video/_media", "format": "mp4",
        "disable_caching": True, "verbosity": "WARNING",
    }):
        Scene01ExocentricWorldlines().render()
        print("rendered:", config.output_file, f"({w}x{h}@{fps})")

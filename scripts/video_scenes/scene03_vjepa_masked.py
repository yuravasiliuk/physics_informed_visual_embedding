"""Scene 03 — V-JEPA 2 core concept: predict masked video in latent space.

Curtain-raiser for the linear-probe scenes. Message, made visual:
  * A video clip is cut into spacetime patches (16px, tubelet 2).
  * MOST patches are masked (hidden).
  * A CONTEXT encoder (ViT-g: 40 transformer blocks, 1408-d) sees only the
    VISIBLE patches; a PREDICTOR guesses the representations of the hidden ones.
  * The training signal lives in REPRESENTATION SPACE, compared against a
    TARGET encoder (EMA weights) — predict features, never pixels. (This
    masked-only, latent-space loss is exactly what the probe scenes interrogate.)
  * Sequel hook: we FREEZE all 40 blocks and read each with a linear probe to
    ask which layers linearly expose physics.

Numbers are the report's ViT-g variant (reports/2026-07-06-frozen-vjepa2-physics-
probe.md §4): 40 blocks · 1408-d · 384 px in · patch 16 / tubelet 2.

Style follows scene 01/02 (manim 0.20.1, teacher palette). Self-contained:
    python scripts/video_scenes/scene03_vjepa_masked.py            # quick preview
    python scripts/video_scenes/scene03_vjepa_masked.py -q h       # 1080p
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from manim import (
    Scene, Rectangle, RoundedRectangle, Square, VGroup, Text, Arrow, Line, Dot,
    Brace, FadeIn, FadeOut, Create, GrowArrow, config, tempconfig, smooth,
    ORIGIN, UP, DOWN, LEFT, RIGHT,
)

# Reuse the palette from scene 01 (importing it is side-effect free — scene 01
# only renders under its own __main__). sys.path insert makes this robust to cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene01_allocentric_worldlines import (
    BG, TEXT, MUTED, CYAN, GREEN, YELLOW, MAGENTA,
)

# ------------------------------- clip geometry ------------------------------
NCOL, NROW = 5, 4
PS = 0.32                                  # patch square side
VISIBLE = {2, 5, 8, 11, 13, 16, 19}        # which patch indices survive the mask


def _fit(mob, frac: float = 0.92):
    """Scale a mobject down so it never overflows the frame width."""
    if mob.width > config.frame_width * frac:
        mob.scale(config.frame_width * frac / mob.width)
    return mob


def build_clip():
    """A grid of spacetime patches with a couple of ghost frame outlines behind
    it (to read as *video* = time). Returns (group, list_of_patch_squares)."""
    squares = []
    grid = VGroup()
    for r in range(NROW):
        for c in range(NCOL):
            sq = Square(PS, stroke_color=MUTED, stroke_width=1.1,
                        fill_color=CYAN, fill_opacity=0.85)
            sq.move_to([(c - (NCOL - 1) / 2) * PS, ((NROW - 1) / 2 - r) * PS, 0])
            squares.append(sq)
            grid.add(sq)
    ghosts = VGroup()
    for k in (2, 1):                        # offset outlines = earlier frames
        g = Rectangle(width=NCOL * PS + 0.05, height=NROW * PS + 0.05,
                      stroke_color=MUTED, stroke_width=1.0)
        g.set_stroke(opacity=0.28).shift(0.11 * k * (RIGHT + UP)).set_z_index(-k)
        ghosts.add(g)
    return VGroup(ghosts, grid), squares


def tower_separators(tower: RoundedRectangle, n: int = 9) -> VGroup:
    """Faint horizontal rules inside the encoder tower (evoke stacked blocks)."""
    x = tower.get_x()
    hw = tower.width / 2 - 0.12
    ys = np.linspace(tower.get_bottom()[1] + 0.24, tower.get_top()[1] - 0.24, n)
    seps = VGroup(*[
        Line([x - hw, y, 0], [x + hw, y, 0], stroke_color=CYAN, stroke_width=1.0)
        .set_stroke(opacity=0.28) for y in ys
    ])
    return seps


def token_row(color: str, center, n: int = 6) -> VGroup:
    row = VGroup(*[Square(0.2, stroke_width=0, fill_color=color, fill_opacity=0.9)
                   for _ in range(n)]).arrange(RIGHT, buff=0.09)
    row.move_to(center)
    return row


# --------------------------------- the scene --------------------------------
class Scene03VJepaMasked(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("V-JEPA 2 — predicting masked video", font_size=38,
                     color=TEXT, weight="BOLD").to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=0.2 * DOWN, run_time=0.6))

        # ---- 1. the clip -------------------------------------------------
        clip, squares = build_clip()
        clip.move_to([-4.7, 0.35, 0])
        clab = Text("video clip · spacetime patches", font_size=18, color=MUTED)
        clab.next_to(clip, DOWN, buff=0.22)
        subcap = Text("16 px patches · 2-frame tubelets\n384×384 px input", font_size=14,
                      color=MUTED, line_spacing=0.6)
        subcap.next_to(clab, DOWN, buff=0.08)
        self.play(FadeIn(clip, run_time=0.7), FadeIn(clab, run_time=0.5))
        self.play(FadeIn(subcap, run_time=0.35))
        self.wait(0.2)

        # ---- 2. mask most patches ---------------------------------------
        mask_lbl = Text("mask most patches", font_size=21, color=YELLOW,
                        weight="BOLD").next_to(clip, UP, buff=0.28)
        hide = [squares[i].animate.set_fill(BG, opacity=1.0)
                .set_stroke(MUTED, width=1.0, opacity=0.45)
                for i in range(len(squares)) if i not in VISIBLE]
        self.play(*hide, FadeIn(mask_lbl, shift=0.12 * DOWN), run_time=0.9)
        self.play(*[squares[i].animate.set_fill(CYAN, opacity=1.0) for i in VISIBLE],
                  run_time=0.35)
        self.wait(0.2)

        # ---- 3. context encoder tower (the 40 blocks) -------------------
        tower = RoundedRectangle(width=1.5, height=2.4, corner_radius=0.12,
                                 stroke_color=CYAN, stroke_width=2.5,
                                 fill_color=CYAN, fill_opacity=0.06)
        tower.move_to([-2.3, -0.7, 0])
        seps = tower_separators(tower)
        enc_title = Text("context encoder · ViT-g", font_size=19, color=CYAN,
                         weight="BOLD").next_to(tower, UP, buff=0.18)
        blocks_lbl = Text("40 transformer blocks · 1408-d", font_size=20, color=TEXT,
                          weight="BOLD").next_to(tower, DOWN, buff=0.22)
        a_ctx = Arrow(clip.get_right() + 0.05 * RIGHT, tower.get_left() + 0.05 * LEFT,
                      buff=0.12, color=CYAN, stroke_width=4,
                      max_tip_length_to_length_ratio=0.14)
        a_ctx_lbl = Text("visible only", font_size=16, color=CYAN).next_to(
            a_ctx, UP, buff=0.06).shift(0.15 * RIGHT)
        self.play(Create(tower), FadeIn(seps), FadeIn(enc_title),
                  GrowArrow(a_ctx), FadeIn(a_ctx_lbl), run_time=0.9)
        self.play(FadeIn(blocks_lbl, shift=0.1 * UP), run_time=0.5)
        # depth sweep: a highlight rides up through the blocks
        sweep = Rectangle(width=tower.width - 0.08, height=0.26, stroke_width=0,
                          fill_color=CYAN, fill_opacity=0.5)
        sweep.move_to([tower.get_x(), tower.get_bottom()[1] + 0.25, 0])
        self.add(sweep)
        self.play(sweep.animate.move_to([tower.get_x(), tower.get_top()[1] - 0.25, 0]),
                  run_time=1.2, rate_func=smooth)
        self.play(FadeOut(sweep, run_time=0.25))
        self.wait(0.15)

        # ---- 4. target encoder branch (EMA) -----------------------------
        tenc = RoundedRectangle(width=1.6, height=1.0, corner_radius=0.12,
                                stroke_color=GREEN, stroke_width=2.5,
                                fill_color=GREEN, fill_opacity=0.07)
        tenc.move_to([-2.3, 2.15, 0])
        tenc_lbl = Text("target encoder\n(EMA weights)", font_size=16, color=GREEN,
                        weight="BOLD", line_spacing=0.7).move_to(tenc.get_center())
        a_full = Arrow(clip.get_top() + 0.1 * UP, tenc.get_left() + 0.05 * LEFT,
                       buff=0.12, color=GREEN, stroke_width=3.5,
                       max_tip_length_to_length_ratio=0.12)
        a_full_lbl = Text("full clip", font_size=16, color=GREEN).next_to(
            a_full, UP, buff=0.04).shift(0.2 * LEFT)
        self.play(Create(tenc), FadeIn(tenc_lbl), GrowArrow(a_full),
                  FadeIn(a_full_lbl), run_time=0.8)

        # ---- 5. predictor -----------------------------------------------
        pred = RoundedRectangle(width=1.5, height=1.0, corner_radius=0.12,
                                stroke_color=MAGENTA, stroke_width=2.5,
                                fill_color=MAGENTA, fill_opacity=0.08)
        pred.move_to([0.5, -0.7, 0])
        pred_lbl = Text("predictor", font_size=19, color=MAGENTA,
                        weight="BOLD").move_to(pred.get_center())
        a_pred = Arrow(tower.get_right() + 0.05 * RIGHT, pred.get_left() + 0.05 * LEFT,
                       buff=0.1, color=CYAN, stroke_width=4,
                       max_tip_length_to_length_ratio=0.16)
        self.play(Create(pred), FadeIn(pred_lbl), GrowArrow(a_pred), run_time=0.7)

        # ---- 6. latent tokens + loss ------------------------------------
        pred_tok = token_row(MAGENTA, [2.7, -0.7, 0])
        targ_tok = token_row(GREEN, [2.7, 2.15, 0])
        a_pt = Arrow(pred.get_right() + 0.05 * RIGHT, pred_tok.get_left() + 0.05 * LEFT,
                     buff=0.1, color=MAGENTA, stroke_width=3.5,
                     max_tip_length_to_length_ratio=0.18)
        a_tt = Arrow(tenc.get_right() + 0.05 * RIGHT, targ_tok.get_left() + 0.05 * LEFT,
                     buff=0.1, color=GREEN, stroke_width=3.5,
                     max_tip_length_to_length_ratio=0.1)
        approx = Text("≈", font_size=40, color=YELLOW).move_to([2.7, 0.72, 0])
        loss_brace = Brace(VGroup(pred_tok, targ_tok), RIGHT, color=YELLOW)
        loss_lbl = Text("loss in\nlatent space", font_size=19, color=YELLOW,
                        weight="BOLD", line_spacing=0.7).next_to(loss_brace, RIGHT, buff=0.12)
        self.play(GrowArrow(a_pt), GrowArrow(a_tt),
                  FadeIn(pred_tok), FadeIn(targ_tok), run_time=0.7)
        self.play(FadeIn(approx), Create(loss_brace), FadeIn(loss_lbl), run_time=0.6)
        core = _fit(Text("predict the features of the hidden patches — never their pixels",
                         font_size=23, color=TEXT)).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(core, shift=0.12 * UP), run_time=0.6)
        self.wait(0.9)

        # ---- 7. sequel hook: freeze the encoder, probe every block ------
        machinery = VGroup(tenc, tenc_lbl, a_full, a_full_lbl, pred, pred_lbl,
                           a_pred, pred_tok, targ_tok, a_pt, a_tt, approx,
                           loss_brace, loss_lbl)
        self.play(machinery.animate.set_opacity(0.12), FadeOut(core),
                  FadeOut(mask_lbl), FadeOut(a_ctx_lbl), run_time=0.7)

        frozen = Text("frozen — no gradients", font_size=19, color=CYAN,
                      weight="BOLD").next_to(enc_title, UP, buff=0.12)
        # probe taps off the mid-to-late blocks (28–32 of 40 → upper portion)
        taps = VGroup()
        for y in np.linspace(-0.35, 0.45, 5):
            t = Arrow([tower.get_right()[0] + 0.02, y, 0],
                      [tower.get_right()[0] + 0.7, y, 0], buff=0.0, color=YELLOW,
                      stroke_width=3, max_tip_length_to_length_ratio=0.35)
            taps.add(t, Dot([tower.get_right()[0] + 0.85, y, 0], radius=0.05, color=YELLOW))
        taps_lbl = Text("linear probe\ntaps each block", font_size=17, color=YELLOW,
                        weight="BOLD", line_spacing=0.7).next_to(taps, RIGHT, buff=0.18)
        self.play(FadeIn(frozen, shift=0.1 * DOWN), run_time=0.5)
        self.play(*[GrowArrow(m) for m in taps if isinstance(m, Arrow)],
                  *[FadeIn(m) for m in taps if isinstance(m, Dot)],
                  FadeIn(taps_lbl), run_time=0.9)

        preview = _fit(Text("Next: freeze all 40 blocks and read each with a linear "
                            "probe — which layers expose physics?",
                            font_size=22, color=TEXT), frac=0.9).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(preview, shift=0.12 * UP), run_time=0.7)
        self.wait(1.4)


# --------------------------------- rendering --------------------------------
_QUALITY = {"l": (854, 480, 15), "m": (1280, 720, 30), "h": (1920, 1080, 30)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", choices=list(_QUALITY), default="l")
    ap.add_argument("--out", default="scene03_vjepa_masked")
    args = ap.parse_args()
    w, h, fps = _QUALITY[args.quality]
    with tempconfig({
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "background_color": BG, "output_file": args.out,
        "media_dir": "reports/video/_media", "format": "mp4",
        "disable_caching": True, "verbosity": "WARNING",
    }):
        Scene03VJepaMasked().render()
        print("rendered:", config.output_file, f"({w}x{h}@{fps})")

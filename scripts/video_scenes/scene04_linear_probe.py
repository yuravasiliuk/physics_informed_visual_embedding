"""Scene 04 — How a linear probe reads a frozen encoder (intern-friendly).

Re-cut 2026-07-16 to the recorded TTS voiceover
`reports/video/narration/scene04_linear_probe.wav` (73.88 s; speech ends
72.74 s). Unlike scene 03b the bottom caption bar STAYS — these visuals are
schematic, so the captions double as subtitles — but every caption now mirrors
the recorded sentence (lightly copyedited: "a weight", digits for the spoken
numbers), and every beat is cued to the wav's pause boundaries (ffmpeg
silencedetect, noise=-32dB:d=0.35) on the *absolute* scene clock via
`until(t)`, so transform durations can't shift later beats.

Changes vs the old captioned 44 s cut, following the recording:
  * the standalone "a linear probe is just a weighted sum" beat is not
    narrated — the weighted-sum machinery (w column, Σ, multiply-add fly-ins,
    formula) now builds during "Multiply each feature by a weight, add a
    bias…";
  * the score-definition beat (scatter plot + `score = 0.60` readout) is not
    narrated and was DROPPED — "score" first appears on the profile's y-axis,
    and scene 05 re-introduces the 0.60 number;
  * the two fit-convergence steps (ŷ walking toward the truth) moved under
    "The probe reads whether there is a linear dependency…".
  * NOTE: the recording says "nine thousands sixteen" but the true token
    count is 9216 (= 16 × 24 × 24) — the screen shows the correct number. If
    the TTS is ever regenerated, fix the line to "nine thousand two hundred
    sixteen" (and "by a weights" → "by a weight"), then re-check the cues.

Everything on screen is still grounded in the actual protocol of
reports/2026-07-06-frozen-vjepa2-physics-probe.md §3:

  * Encoder = frozen V-JEPA 2 (40 transformer blocks, 1408-d tokens) — never trained.
  * A layer's raw output is NOT one vector: it is a token field of
    16 × 24 × 24 = 9216 space-time tokens (32 frames @ 384², tubelet 2, patch 16),
    each 1408-d — see plans/probe/vjepa-embedding-structure.md §1. The probe input
    is the mask-weighted average of the tokens under the object
    (encoders.py:encode_pooled) → a single 1408-d vector h. So "1408" is one
    token's width, not the size of the whole embedding.
  * A *linear* probe is a ridge regression: ŷ = w·h + b (a weighted sum + bias,
    no hidden layers, no nonlinearity). Only w, b are fit; the encoder never moves.
  * Score = R² on held-out clips (1 perfect, 0 = predict-the-mean). On screen
    the metric is called just "score" — R² is never named (lay audience).
  * Repeat per block → a profile; velocity peaks at blocks 28–32 of 40.

    python scripts/video_scenes/scene04_linear_probe.py            # quick preview
    python scripts/video_scenes/scene04_linear_probe.py -q h       # 1080p
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from manim import (
    Scene, VGroup, Text, RoundedRectangle, Rectangle, Square, Circle, Arc,
    Triangle, Line, DashedLine, Arrow, CurvedArrow, Dot, Axes,
    FadeIn, FadeOut, Transform, ReplacementTransform, Create, Write, GrowArrow,
    Indicate, Flash,
    LaggedStart, always_redraw, ValueTracker, config, tempconfig, smooth,
    UP, DOWN, LEFT, RIGHT, PI,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene01_allocentric_worldlines import (
    BG, TEXT, MUTED, GRID, CYAN, GREEN, MAGENTA, YELLOW, RED, BLUE,
)

N_BLK = 12          # drawn blocks (schematic stand-in for the real 40)
BY = 2.62           # y of the encoder row


def _fit(mob, frac: float = 0.94):
    if mob.width > config.frame_width * frac:
        mob.scale(config.frame_width * frac / mob.width)
    return mob


# fixed "values" so a feature/weight column looks like real numbers, not a bar
_SH_H = [0.72, 0.30, 0.58, 0.24]
_SH_W = [0.35, 0.66, 0.22, 0.50]


def _cell_column(shades, color, cell: float = 0.34) -> VGroup:
    """A vector drawn as a stack of value-shaded cells + ⋮ + one tail cell."""
    top = VGroup(*[
        Square(cell, stroke_color=color, stroke_width=2.0,
               fill_color=color, fill_opacity=op)
        for op in shades
    ]).arrange(DOWN, buff=0.06)
    dots = Text("⋮", font_size=22, color=color)
    tail = Square(cell, stroke_color=color, stroke_width=2.0,
                  fill_color=color, fill_opacity=0.48)
    return VGroup(top, dots, tail).arrange(DOWN, buff=0.08)


def _rows(col: VGroup) -> list:
    """The individual cell mobjects of a _cell_column (top cells + tail)."""
    return list(col[0]) + [col[2]]


def _lock(h: float = 0.32, color=MUTED) -> VGroup:
    body = RoundedRectangle(width=h * 0.92, height=h * 0.72, corner_radius=0.04,
                            stroke_color=color, stroke_width=2.4,
                            fill_color=color, fill_opacity=0.20)
    shackle = Arc(radius=h * 0.30, start_angle=0, angle=PI,
                  stroke_color=color, stroke_width=2.4)
    shackle.next_to(body, UP, buff=-0.03)
    return VGroup(shackle, body)


def _clip_icon() -> VGroup:
    box = RoundedRectangle(width=0.95, height=0.78, corner_radius=0.08,
                           stroke_color=CYAN, stroke_width=2.4,
                           fill_color=BG, fill_opacity=1.0)
    div = VGroup(*[Line([x, -0.30, 0], [x, 0.30, 0], stroke_color=CYAN,
                        stroke_width=1.2, stroke_opacity=0.5) for x in (-0.16, 0.16)])
    tri = Triangle().scale(0.12).set_fill(CYAN, 1).set_stroke(width=0).rotate(-PI / 2)
    return VGroup(box, div, tri)


def _formula() -> VGroup:
    parts = [
        Text("ŷ", font_size=32, color=MAGENTA, weight="BOLD"),
        Text("=", font_size=32, color=TEXT),
        Text("w", font_size=32, color=YELLOW, weight="BOLD"),
        Text("·", font_size=32, color=TEXT),
        Text("hₗ", font_size=32, color=CYAN, weight="BOLD"),
        Text("+  b", font_size=32, color=YELLOW),
    ]
    return VGroup(*parts).arrange(RIGHT, buff=0.16)


def r2_of_layer(L: float) -> float:
    """Schematic velocity-probe profile: rises, peaks ~block 30, plateaus (§4)."""
    x = (L - 1) / 39.0
    base = 0.62 / (1.0 + np.exp(-(x - 0.45) * 11.0))
    peak = 0.06 * np.exp(-((x - 0.76) / 0.12) ** 2)
    return float(base + peak)


class Scene04LinearProbe(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.narr = None

        def until(t: float):
            """Sleep until absolute scene time t (the wav timeline)."""
            dt = t - self.renderer.time
            if dt > 0:
                self.wait(dt)

        # ---------------- persistent frozen encoder (top) ------------------
        # VO 0.0–3.0: "It is important to understand what is linear probe."
        title = _fit(Text("What is a linear probe?", font_size=36, color=TEXT,
                          weight="BOLD")).to_edge(UP, buff=0.30)
        blocks = VGroup(*[
            RoundedRectangle(width=0.5, height=0.86, corner_radius=0.08,
                             stroke_color=BLUE, stroke_width=2.2,
                             fill_color=BLUE, fill_opacity=0.14)
            for _ in range(N_BLK)
        ]).arrange(RIGHT, buff=0.12).move_to([-0.15, BY, 0])
        clip = _clip_icon().next_to(blocks, LEFT, buff=0.5)
        clip_lbl = Text("video clip", font_size=17, color=MUTED).next_to(clip, DOWN, buff=0.14)
        enc_lbl = VGroup(
            _lock(0.34),
            Text("V-JEPA 2  ·  40 frozen transformer blocks  ·  1408-d tokens",
                 font_size=19, color=MUTED),
        ).arrange(RIGHT, buff=0.18).next_to(blocks, DOWN, buff=0.26)

        self._say("It is important to understand what a linear probe is.")
        self.play(FadeIn(title, shift=0.12 * DOWN, run_time=0.6))
        self.play(FadeIn(clip, run_time=0.4), FadeIn(clip_lbl, run_time=0.4),
                  LaggedStart(*[FadeIn(b, shift=0.1 * DOWN) for b in blocks],
                              lag_ratio=0.06, run_time=1.0))
        self.play(FadeIn(enc_lbl, run_time=0.5))

        # Beat 2 — VO 3.6–10.2: "One forward pass through frozen encoder and
        # every block turns video clip into latent representation."
        until(3.55)
        self._say("One forward pass through the frozen encoder — every block\n"
                  "turns the video clip into a latent representation.")
        self.play(LaggedStart(*[Indicate(b, color=CYAN, scale_factor=1.14) for b in blocks],
                              lag_ratio=0.12, run_time=2.2))

        # Beat 3 — VO 11.2–20.7: "Let's pick one layer: its output is not one
        # vector — it's a field of [9216] space-time tokens."  (the recording
        # misspeaks the number; the screen shows the true 9216 = 16·24·24)
        until(11.15)
        pick = blocks[8]
        halo = RoundedRectangle(width=0.62, height=0.98, corner_radius=0.08,
                                stroke_color=CYAN, stroke_width=3.5).move_to(pick)
        self._say("Pick one layer: its output is not one vector —\n"
                  "it's a field of 9216 space-time tokens.")
        # front 6×6 grid stands in for one 24×24 spatial slice; two ghost copies
        # behind suggest the 16-step time axis of the token volume
        g_rng = np.random.default_rng(11)
        grid = VGroup(*[
            Square(0.17, stroke_color=CYAN, stroke_width=1.1,
                   fill_color=CYAN, fill_opacity=float(op))
            for op in g_rng.uniform(0.08, 0.40, 36)
        ]).arrange_in_grid(rows=6, cols=6, buff=0.035)
        ghosts = VGroup(*[
            grid.copy().set_stroke(opacity=0.22).set_fill(opacity=0.05)
                .shift([0.15 * k, 0.15 * k, 0]) for k in (2, 1)
        ])
        field = VGroup(ghosts, grid).move_to([-4.45, -0.42, 0])
        field_lbl = Text("16 time × 24 × 24 = 9216 tokens\neach 1408-d",
                         font_size=15, color=MUTED, line_spacing=0.8)
        field_lbl.next_to(field, DOWN, buff=0.18)
        tap = Arrow(pick.get_bottom(), field.get_top() + 0.12 * UP,
                    color=CYAN, stroke_width=3, buff=0.15, max_tip_length_to_length_ratio=0.08)
        self.play(Create(halo, run_time=0.5))
        self.play(GrowArrow(tap, run_time=0.6), FadeIn(field, shift=0.2 * DOWN, run_time=0.7))
        self.play(FadeIn(field_lbl, run_time=0.4))

        # Beat 4 — VO 21.75–28.0: "Average the tokens under the object's mask
        # into a single [1408]-d vector."  mask-pool: light up the object's
        # tokens, dim the rest, collapse → hₗ
        until(21.7)
        self._say("Average the tokens under the object's mask\n"
                  "into a single 1408-d vector.")
        masked_idx = (9, 14, 15, 16, 20, 21, 22)
        masked = VGroup(*[grid[i] for i in masked_idx])
        others = VGroup(*[grid[i] for i in range(36) if i not in masked_idx])
        mask_lbl = Text("object's\ntokens", font_size=14, color=GREEN,
                        line_spacing=0.8).next_to(field, RIGHT, buff=0.22)
        self.play(masked.animate.set_fill(GREEN, opacity=0.55).set_stroke(GREEN),
                  others.animate.set_opacity(0.15),
                  FadeIn(mask_lbl, run_time=0.4), run_time=0.7)
        self.wait(0.5)
        hcol = _cell_column(_SH_H, CYAN).move_to([-4.55, -0.7, 0])
        h_lbl = Text("hₗ  (1408-d)\nmask-pooled", font_size=18, color=CYAN,
                     line_spacing=0.7).next_to(hcol, DOWN, buff=0.20)
        self.play(FadeOut(tap, run_time=0.3), FadeOut(field_lbl, run_time=0.3),
                  FadeOut(mask_lbl, run_time=0.3),
                  ReplacementTransform(field, hcol, run_time=0.9))
        self.play(FadeIn(h_lbl, run_time=0.4))

        # Beat 5 — VO 28.4–37.1: "Multiply each feature by a weight, add a
        # bias — one prediction, no hidden layers, no nonlinearity. That's the
        # whole model."  The whole weighted-sum machinery builds here (the old
        # standalone "just a weighted sum" beat is not narrated).
        until(28.4)
        self._say("Multiply each feature by a weight, add a bias — one prediction.\n"
                  "No hidden layers, no nonlinearity. That's the whole model.")
        formula = _formula().move_to([-0.6, 1.32, 0])
        wcol = _cell_column(_SH_W, YELLOW).move_to([-2.75, -0.7, 0])
        w_lbl = Text("w  (the probe)", font_size=18, color=YELLOW).next_to(wcol, DOWN, buff=0.22)
        times = Text("×", font_size=30, color=TEXT).move_to(
            [(hcol.get_center()[0] + wcol.get_center()[0]) / 2, -0.7, 0])
        self.play(FadeIn(wcol, shift=0.2 * DOWN, run_time=0.6),
                  FadeIn(w_lbl, run_time=0.4), Write(times, run_time=0.3))

        sigma = Circle(radius=0.42, stroke_color=MAGENTA, stroke_width=3.5,
                       fill_color=MAGENTA, fill_opacity=0.12).move_to([-1.15, -0.7, 0])
        sig_txt = Text("Σ", font_size=34, color=MAGENTA, weight="BOLD").move_to(sigma)
        into = Arrow(wcol.get_right(), sigma.get_left(), color=MUTED,
                     stroke_width=3, buff=0.18, max_tip_length_to_length_ratio=0.12)
        self.play(GrowArrow(into, run_time=0.4), FadeIn(sigma, run_time=0.4),
                  Write(sig_txt, run_time=0.3))

        # multiply-add: each paired row flies into Σ
        hr, wr = _rows(hcol), _rows(wcol)
        sc = sigma.get_center()
        for i in range(len(hr)):
            d = Dot(color=MAGENTA, radius=0.05).move_to(
                (hr[i].get_center() + wr[i].get_center()) / 2)
            self.play(Indicate(hr[i], color=CYAN, scale_factor=1.25),
                      Indicate(wr[i], color=YELLOW, scale_factor=1.25), run_time=0.16)
            self.add(d)
            self.play(d.animate.move_to(sc).set_opacity(0.0), run_time=0.2)
            self.remove(d)

        yhat = VGroup(
            RoundedRectangle(width=1.9, height=0.72, corner_radius=0.1,
                             stroke_color=MAGENTA, stroke_width=3,
                             fill_color=MAGENTA, fill_opacity=0.16),
            Text("ŷ = 1.6 m/s", font_size=22, color=TEXT),
        ).move_to([1.15, -0.7, 0])
        out = Arrow(sigma.get_right(), yhat.get_left(), color=MAGENTA,
                    stroke_width=3, buff=0.15, max_tip_length_to_length_ratio=0.14)
        self.play(Flash(sigma, color=MAGENTA, line_length=0.2, num_lines=12, run_time=0.5),
                  GrowArrow(out, run_time=0.4), FadeIn(yhat, run_time=0.5))
        self.play(Write(formula, run_time=0.9))

        # Beat 6 — VO 37.6–42.4: "We fit ONLY these weights to the labels —
        # the encoder never retrained."
        until(37.55)
        self._say("We fit ONLY these weights to the labels —\n"
                  "the encoder is never retrained.", color=YELLOW)
        truth = VGroup(
            RoundedRectangle(width=2.05, height=0.72, corner_radius=0.1,
                             stroke_color=GREEN, stroke_width=3,
                             fill_color=GREEN, fill_opacity=0.14),
            Text("truth  v = 1.8 m/s", font_size=20, color=TEXT),
        ).move_to([3.7, -0.7, 0])
        err = DashedLine(yhat.get_right(), truth.get_left(), color=RED,
                         stroke_width=3, dash_length=0.09)
        err_lbl = Text("error", font_size=15, color=RED).next_to(err, UP, buff=0.06)
        self.play(FadeIn(truth, run_time=0.5))
        self.play(Create(err, run_time=0.4), FadeIn(err_lbl, run_time=0.3))

        # gradient goes to w (yes) but not into the encoder (no)
        g_yes = CurvedArrow(err.get_center() + 0.1 * DOWN, wcol.get_bottom() + 0.15 * DOWN,
                            color=YELLOW, stroke_width=3, angle=-0.9)
        g_no = DashedLine(err.get_center() + 0.1 * UP, [pick.get_bottom()[0], 1.7, 0],
                          color=RED, stroke_width=2.5, dash_length=0.1)
        no_x = Text("✕  frozen", font_size=18, color=RED).move_to(
            [pick.get_bottom()[0] + 0.9, 1.55, 0])
        self.play(Create(g_yes, run_time=0.5), Create(g_no, run_time=0.5), FadeIn(no_x, run_time=0.3))
        self.play(Indicate(enc_lbl[0], color=MUTED, scale_factor=1.3, run_time=0.6))

        # Beat 7 — VO 42.9–49.1: "The probe reads if there is linear
        # dependency between intermediate feature layers and labels."
        # A couple of "fit" steps make the dependency visible: weights flash,
        # ŷ walks toward the truth, the error shrinks.
        until(42.85)
        self._say("The probe reads whether there is a linear dependency\n"
                  "between intermediate feature layers and the labels.", color=YELLOW)
        for val, frac in ((1.72, 0.55), (1.81, 0.12)):
            new_txt = Text(f"ŷ = {val} m/s", font_size=22, color=TEXT).move_to(yhat[1])
            new_err = DashedLine(yhat.get_right(),
                                 yhat.get_right() + (truth.get_left() - yhat.get_right()) * frac,
                                 color=RED, stroke_width=3, dash_length=0.09)
            self.play(Indicate(wcol, color=YELLOW, scale_factor=1.06),
                      Transform(yhat[1], new_txt),
                      Transform(err, new_err), run_time=0.7)

        # clear the mechanism during the 49.1–50.1 pause before "We do it…"
        until(49.1)
        mech = VGroup(hcol, h_lbl, wcol, w_lbl, times, sigma, sig_txt, into, out,
                      yhat, truth, err, err_lbl, formula, g_yes, g_no, no_x, halo)
        self.play(FadeOut(mech, run_time=0.7))

        # Beat 8 — VO 50.1–55.0: "We do it for every transformer block — 40
        # probes, 40 scores."  (the old score-definition scatter beat is not
        # narrated and was dropped — see the module docstring)
        until(50.05)
        self._say("We do it for every transformer block — 40 probes, 40 scores.")
        lax = Axes(x_range=[0, 40, 10], y_range=[0, 0.8, 0.2], x_length=6.4, y_length=2.5,
                   tips=False, axis_config=dict(stroke_color=MUTED, stroke_width=2))
        lax.move_to([-0.15, -1.35, 0])
        lxl = Text("transformer block   1 → 40", font_size=18, color=MUTED).next_to(lax, DOWN, buff=0.18)
        lyl = Text("score  (velocity)", font_size=18, color=GREEN).next_to(lax, LEFT, buff=0.12).shift(0.1 * UP)
        self.play(Create(lax, run_time=0.7), FadeIn(lxl, run_time=0.3), FadeIn(lyl, run_time=0.3))

        Lt = ValueTracker(1.0)
        x_l, x_r = blocks[0].get_center()[0], blocks[-1].get_center()[0]
        hi = RoundedRectangle(width=0.62, height=0.98, corner_radius=0.08,
                              stroke_color=YELLOW, stroke_width=3.5,
                              fill_color=YELLOW, fill_opacity=0.12).move_to([x_l, BY, 0])
        hi.add_updater(lambda m: m.move_to(
            [x_l + (x_r - x_l) * (Lt.get_value() - 1) / 39.0, BY, 0]))
        curve = always_redraw(lambda: lax.plot(
            r2_of_layer, x_range=[1, max(1.02, Lt.get_value())], color=GREEN, stroke_width=4))
        head = always_redraw(lambda: Dot(
            lax.c2p(Lt.get_value(), r2_of_layer(Lt.get_value())), radius=0.06, color=GREEN))
        guide = always_redraw(lambda: DashedLine(
            hi.get_bottom(), lax.c2p(Lt.get_value(), r2_of_layer(Lt.get_value())),
            color=MUTED, stroke_width=1.5, dash_length=0.08).set_opacity(0.45))
        self.add(hi, guide, curve, head)
        self.play(Lt.animate.set_value(40.0), run_time=4.0, rate_func=smooth)
        hi.clear_updaters()

        # Beat 9 — VO 55.4–66.8: "The profile shows WHERE the signal lives —
        # which layers a real head should tap."
        until(55.4)
        # single line — a second line would collide with the chart's x label
        self._say("The profile shows WHERE the signal lives — which layers a real head should tap.")
        peak_x = lax.c2p(30, r2_of_layer(30))
        peak = Dot(peak_x, radius=0.08, color=YELLOW)
        peak_lbl = Text("peaks at blocks 28–32", font_size=19, color=YELLOW).next_to(peak, UP, buff=0.18)
        self.play(FadeIn(peak, scale=0.5), Flash(peak_x, color=YELLOW, line_length=0.15, run_time=0.5),
                  FadeIn(peak_lbl, run_time=0.4))

        # ------- closing handoff — VO 67.6–72.7: "the layer-by-layer -------
        # profiles for real physics — velocity, spin, and center of mass."
        # cleared during the 66.8–67.6 pause; the card mirrors the sentence.
        until(66.5)
        self.play(FadeOut(VGroup(hi, guide, curve, head, peak, peak_lbl, lax, lxl, lyl),
                          run_time=0.6),
                  FadeOut(self.narr, run_time=0.4))
        closing = _fit(Text("Next: the layer-by-layer profiles for real physics —\n"
                            "velocity, spin, and center of mass.",
                            font_size=28, color=TEXT, weight="BOLD", line_spacing=0.8), 0.9)
        closing.move_to([0, -0.2, 0])
        self.play(FadeIn(closing, shift=0.12 * UP, run_time=0.8))
        until(74.0)    # wav is 73.88 s — outlast it slightly so the mux never pads

    # -------- narration bar (mirrors the sibling scenes' cross-fade) --------
    def _say(self, text, color=TEXT):
        new = _fit(Text(text, font_size=22, color=color, line_spacing=0.7), 0.94)
        new.to_edge(DOWN, buff=0.34)
        if self.narr is None:
            self.narr = new
            self.play(FadeIn(new, shift=0.1 * UP), run_time=0.4)
        else:
            self.play(FadeOut(self.narr, run_time=0.25))
            self.narr = new
            self.play(FadeIn(new, shift=0.1 * UP), run_time=0.4)


# --------------------------------- rendering --------------------------------
_QUALITY = {"l": (854, 480, 15), "m": (1280, 720, 30), "h": (1920, 1080, 30)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", choices=list(_QUALITY), default="l")
    ap.add_argument("--out", default="scene04_linear_probe")
    args = ap.parse_args()
    w, h, fps = _QUALITY[args.quality]
    with tempconfig({
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "background_color": BG, "output_file": args.out,
        "media_dir": "reports/video/_media", "format": "mp4",
        "disable_caching": True, "verbosity": "WARNING",
    }):
        Scene04LinearProbe().render()
        print("rendered:", config.output_file, f"({w}x{h}@{fps})")

"""Scene 05 — What is the oracle? (perfect physics in place of encoder features)

Sequel to scene 04, now **paced to the recorded voiceover**
`reports/video/narration/scene05_oracle.wav` (68.54 s). Every beat is cued to the
wav's silence boundaries (ffmpeg silencedetect, noise=-32dB:d=0.35) on the
*absolute* scene clock — the `until(t)` helper in construct() — so a long
transform can't shift later beats (same pattern as scenes 03b/04). The bottom
caption bar STAYS (these visuals are schematic, so the captions double as
subtitles), each caption mirroring the recorded sentence, lightly copyedited.

Re-cut 2026-07-16 to the recording — the recorded script is SHORTER than the old
self-paced cut, so several un-narrated beats were dropped, following the recording:
  * the standalone CEILING beat is folded into the "strict superset" line (S7);
  * the two-outcome VERDICT cards (signal present → fix the probe / absent → fix
    the data) are GONE — the recording replaces the whole verdict+fix section with
    a single closing line, "the oracle certifies observability before we probe";
  * the "why it failed — tipped several degrees past balance" toppling diagram is
    GONE (not narrated);
  * the fix beat (balance + eccentric-roll staging, oracle cert lift +0.0199) is
    GONE (not narrated).
  These beats live in git history; if a longer VO is recorded they can be restored.
  The +0.0009 CoM lift and the 0.60 velocity score stay on screen (grounded),
  though the recording states them only qualitatively.

Recorded line onsets (≈ speech start, s):
  S1 0.0 · S2 7.7 · S3 12.3 · S4 20.26 · S5 27.10 · S6 36.28 · S7 38.58 ·
  S8 43.36 · S9 52.10 · S10 58.68 · S11 62.74 (closing card). Wav 68.54 s; the
  video runs to 69.0 s so the mux never pads. S1–S3 share one 19.5 s spoken chunk
  (no pause between), so S2/S3 onsets are word-count estimates, not measured pauses.

Everything on screen is grounded in
reports/2026-07-06-frozen-vjepa2-physics-probe.md §5, §8 and src/probe/com_oracle.py:
  * Velocity probes read R² = 0.60 (on screen just "score" — R² is never named,
    matching scene 04's naming choice); the CoM-offset probe is a flat null.
  * The oracle swaps encoder features for the noiseless GT trajectory: per body,
    per frame, CoM-relative position (3) + tilt (gravity direction in the body
    frame, 3 — com_oracle.py's default since 2026-07-13) + lin-vel (3) + ang-vel
    (3) = 12 × 32 frames = 384 numbers (+ body size & shape). A strict *superset*
    of any motion recoverable from the pixels → it is a CEILING.
  * Actual CoM result: oracle lift +0.0009 ± 0.0023 → zero (report §5). The encoder
    null is "exonerated, not passed": perfect state can't decode it either, so the
    fix is the DATA, not the model — certify observability BEFORE any GPU spend.

Self-contained + cwd-robust like the sibling scenes; reuses scene 04's visual
vocabulary (cell columns, lock, clip icon) so it reads as a direct continuation.
    python scripts/video_scenes/scene05_oracle.py            # quick preview
    python scripts/video_scenes/scene05_oracle.py -q h       # 1080p
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from manim import (
    Scene, VGroup, Text, RoundedRectangle, Rectangle, Square, Circle, Arc,
    Triangle, Line, DashedLine, Arrow, CurvedArrow, Dot, FadeIn, FadeOut,
    Transform, Create, Write, GrowArrow, Indicate, Flash, LaggedStart,
    config, tempconfig, smooth, UP, DOWN, LEFT, RIGHT, PI,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene01_allocentric_worldlines import (
    BG, TEXT, MUTED, GRID, CYAN, GREEN, MAGENTA, YELLOW, RED, BLUE,
)
# reuse scene 04's building blocks so this reads as a direct continuation
from scene04_linear_probe import (
    _fit, _cell_column, _lock, _clip_icon, _SH_H, _SH_W, N_BLK, BY,
)


def _sim_icon(color=GREEN, w: float = 1.2) -> VGroup:
    """A little 'simulator' monitor with a parabolic trajectory inside —
    the ground-truth source (the sim knows the exact physics)."""
    screen = RoundedRectangle(width=w, height=w * 0.75, corner_radius=0.08,
                              stroke_color=color, stroke_width=2.4,
                              fill_color=BG, fill_opacity=1.0)
    xs = np.linspace(-0.38, 0.38, 22) * (w / 1.2)
    ys = (-1.1 * xs ** 2 + 0.16) * (w / 1.2)
    traj = VGroup(*[Line([xs[i], ys[i], 0], [xs[i + 1], ys[i + 1], 0],
                         stroke_color=color, stroke_width=2.0)
                    for i in range(len(xs) - 1)]).move_to(screen)
    ball = Dot(color=color, radius=0.05).move_to(
        screen.get_center() + np.array([xs[-1], ys[-1], 0]) * 0.98)
    return VGroup(screen, traj, ball)


def _probe_box(w: float = 2.3, h: float = 1.05) -> VGroup:
    box = RoundedRectangle(width=w, height=h, corner_radius=0.12,
                           stroke_color=YELLOW, stroke_width=3,
                           fill_color=YELLOW, fill_opacity=0.10)
    inner = VGroup(
        Text("the same linear probe", font_size=17, color=TEXT, weight="BOLD"),
        Text("ŷ = w · x + b", font_size=18, color=YELLOW),
    ).arrange(DOWN, buff=0.12).move_to(box)
    return VGroup(box, inner)


def _chip(text, color, w=3.0, h=0.62, fs=20) -> VGroup:
    box = RoundedRectangle(width=w, height=h, corner_radius=0.1,
                           stroke_color=color, stroke_width=2.6,
                           fill_color=color, fill_opacity=0.12)
    return VGroup(box, Text(text, font_size=fs, color=TEXT).move_to(box))


class Scene05Oracle(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.narr = None

        def until(t: float):
            """Wait until absolute scene time t (the wav timeline)."""
            dt = t - self.renderer.time
            if dt > 0:
                self.wait(dt)

        # ---------------- persistent frozen encoder (top, as in scene 04) ----
        title = _fit(Text("What is the oracle?", font_size=36, color=TEXT,
                          weight="BOLD")).to_edge(UP, buff=0.30)
        blocks = VGroup(*[
            RoundedRectangle(width=0.5, height=0.86, corner_radius=0.08,
                             stroke_color=BLUE, stroke_width=2.2,
                             fill_color=BLUE, fill_opacity=0.14)
            for _ in range(N_BLK)
        ]).arrange(RIGHT, buff=0.12).move_to([-0.15, BY, 0])
        clip = _clip_icon().next_to(blocks, LEFT, buff=0.5)
        enc_lbl = VGroup(
            _lock(0.34),
            Text("V-JEPA 2  ·  40 frozen blocks  ·  1408-d features",
                 font_size=19, color=MUTED),
        ).arrange(RIGHT, buff=0.18).next_to(blocks, DOWN, buff=0.24)

        self.play(FadeIn(title, shift=0.12 * DOWN, run_time=0.4))
        self.play(FadeIn(clip, run_time=0.3),
                  LaggedStart(*[FadeIn(b, shift=0.1 * DOWN) for b in blocks],
                              lag_ratio=0.04, run_time=0.6),
                  FadeIn(enc_lbl, run_time=0.4))

        # ============ B1 — the null with two explanations ====================
        # One continuous 19.5 s spoken chunk: S1 @0.0, S2 @~7.7, S3 @~12.3.
        chip_v = _chip("velocity:  score = 0.60   ✓", GREEN, w=4.1, fs=21
                       ).move_to([0, 1.15, 0])
        self._say("On velocity, the linear probe hits a prediction score of 0.60 —\n"
                  "a clear linear mapping. No surprise.")
        self.play(FadeIn(chip_v, shift=0.1 * UP, run_time=0.5))

        until(7.7)   # S2
        chip_c = _chip("center of mass:  score ≈ 0.00   —  null", RED, fs=21,
                       w=5.9).move_to([0, 0.35, 0])
        self._say("But center of mass — the hidden property —\ncomes back a flat null.")
        self.play(FadeIn(chip_c, shift=0.1 * UP, run_time=0.5))

        until(12.3)  # S3
        qmark = Text("?", font_size=64, color=YELLOW, weight="BOLD").move_to([0, -1.35, 0])
        card1 = VGroup(
            RoundedRectangle(width=3.9, height=0.7, corner_radius=0.1,
                             stroke_color=MUTED, stroke_width=2.2),
            Text("① the encoder is blind to it", font_size=19, color=TEXT),
        )
        card1[1].move_to(card1[0])
        card1.move_to([-3.35, -2.5, 0])
        card2 = VGroup(
            RoundedRectangle(width=3.9, height=0.7, corner_radius=0.1,
                             stroke_color=MUTED, stroke_width=2.2),
            Text("② the data never contained it", font_size=19, color=TEXT),
        )
        card2[1].move_to(card2[0])
        card2.move_to([3.35, -2.5, 0])
        a1 = Arrow(qmark.get_bottom(), card1[0].get_top(), color=MUTED,
                   stroke_width=2.5, buff=0.15, max_tip_length_to_length_ratio=0.1)
        a2 = Arrow(qmark.get_bottom(), card2[0].get_top(), color=MUTED,
                   stroke_width=2.5, buff=0.15, max_tip_length_to_length_ratio=0.1)
        self._say("A null has two explanations: either the encoder is blind to it,\n"
                  "or the data never contained it.")
        self.play(FadeIn(qmark, scale=0.6, run_time=0.5))
        self.play(GrowArrow(a1), GrowArrow(a2),
                  FadeIn(card1, run_time=0.5), FadeIn(card2, run_time=0.5))

        until(19.7)  # clear during the pause before "So we build an oracle…"
        beat1 = VGroup(chip_v, chip_c, qmark, card1, card2, a1, a2)
        self.play(FadeOut(beat1, run_time=0.5))

        # ============ B2 — the oracle: perfect physics as input ==============
        # S4 @20.26 (build) · S5 @27.10 (per-frame fields) · S6 @36.28 (12
        # numbers) · S7 @38.58 (strict superset / ceiling).
        until(20.26)  # S4
        self._say("So we build an oracle: instead of the encoder's features,\n"
                  "take the simulator's own ground-truth physics.")
        sim = _sim_icon(GREEN, w=1.3).move_to([-5.35, 0.15, 0])
        sim_lbl = Text("simulator\n(ground truth)", font_size=16, color=GREEN,
                       line_spacing=0.7).next_to(sim, DOWN, buff=0.16)
        gcol = _cell_column(_SH_W, GREEN, cell=0.30).move_to([-3.15, 0.05, 0])
        feed0 = Arrow(sim.get_right(), gcol.get_left(), color=GREEN, stroke_width=3,
                      buff=0.2, max_tip_length_to_length_ratio=0.12)
        self.play(FadeIn(sim, run_time=0.5), FadeIn(sim_lbl, run_time=0.4))
        self.play(GrowArrow(feed0, run_time=0.4),
                  FadeIn(gcol, shift=0.15 * RIGHT, run_time=0.5))

        until(27.10)  # S5 — the four ground-truth fields, per frame
        legend = VGroup(
            Text("per frame:", font_size=20, color=TEXT, weight="BOLD"),
            Text("position  x y z", font_size=19, color=MUTED),
            Text("tilt  (gravity in body frame)", font_size=19, color=MUTED),
            Text("linear velocity  v", font_size=19, color=MUTED),
            Text("angular velocity  ω  (spin)", font_size=19, color=MUTED),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        rule = Line(LEFT * 2.0, RIGHT * 2.0, stroke_color=MUTED, stroke_width=1.5)
        total = Text("12 numbers per frame  ·  × 32 frames  →  384",
                     font_size=20, color=GREEN, weight="BOLD")
        legblock = VGroup(legend, rule, total).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        legblock.move_to([2.1, 0.35, 0])
        # per-field counts, anchored to a common right column after the move
        cx = legend.get_right()[0] + 0.9
        counts = VGroup(*[Text(c, font_size=19, color=GREEN) for c in ("3", "3", "3", "3")])
        for c, row in zip(counts, legend[1:]):
            c.move_to([cx, row.get_center()[1], 0])
        brace = Arrow(gcol.get_right(), legend.get_left() + 0.1 * LEFT, color=GREEN,
                      stroke_width=2.5, buff=0.2, max_tip_length_to_length_ratio=0.08)
        self._say("For every frame: precise position, tilt against gravity,\n"
                  "velocity, and spin.")
        self.play(GrowArrow(brace, run_time=0.4),
                  FadeIn(legend, run_time=0.6), FadeIn(counts, run_time=0.6))
        self.play(Create(rule, run_time=0.3))

        until(36.28)  # S6 — "Total 12 numbers." (× 32 → 384 shown but not spoken)
        self._say("Twelve numbers in all.")
        self.play(FadeIn(total, shift=0.1 * UP, run_time=0.5))

        until(38.58)  # S7 — the superset / ceiling argument
        superset = Text("noiseless · complete · exact — a strict superset of\n"
                        "any motion an encoder could recover from the pixels",
                        font_size=19, color=CYAN, line_spacing=0.7).move_to([0, -1.7, 0])
        ceil = Text("→  a CEILING on what's knowable", font_size=19, color=GREEN,
                    weight="BOLD").next_to(superset, DOWN, buff=0.16)
        self._say("Noiseless and complete — a strict superset of any motion\n"
                  "an encoder could ever recover from the pixels.")
        self.play(FadeIn(superset, shift=0.1 * UP, run_time=0.6))
        self.play(FadeIn(ceil, run_time=0.4))

        until(42.7)   # clear during the pause before "Now the trick…"
        beat2 = VGroup(sim, sim_lbl, gcol, feed0, legblock, counts, brace, superset, ceil)
        self.play(FadeOut(beat2, run_time=0.5))

        # ============ B3 — the swap: same probe, new input ===================
        # S8 @43.36 (setup) · S9 @52.10 (throw the switch).
        until(43.36)  # S8
        self._say("Now the trick: feed that into the exact same probe.\n"
                  "Same head, same folds, same metric — only the input changes.")
        # input A: encoder features (cyan), tapped from a frozen block
        acol = _cell_column(_SH_H, CYAN, cell=0.24).move_to([-4.8, 0.75, 0])
        a_lbl = Text("encoder features", font_size=15, color=CYAN
                     ).next_to(acol, UP, buff=0.12)
        tapblk = blocks[3]
        tap = Arrow(tapblk.get_bottom(), acol.get_top() + 0.05 * UP, color=CYAN,
                    stroke_width=2.6, buff=0.15, max_tip_length_to_length_ratio=0.06)
        # input B: perfect physics (green), from the simulator
        bcol = _cell_column(_SH_W, GREEN, cell=0.24).move_to([-4.8, -1.5, 0])
        sim2 = _sim_icon(GREEN, w=1.0).move_to([-6.35, -1.5, 0])
        b_lbl = Text("perfect physics · 384-d", font_size=15, color=GREEN
                     ).next_to(bcol, DOWN, buff=0.1)
        feedb = Arrow(sim2.get_right(), bcol.get_left(), color=GREEN, stroke_width=2.4,
                      buff=0.14, max_tip_length_to_length_ratio=0.1)
        pivot = np.array([-2.75, -0.4, 0])
        probe = _probe_box().move_to([-0.35, -0.4, 0])
        wireA = DashedLine(acol.get_right(), pivot, color=MUTED, stroke_width=2,
                           dash_length=0.09)
        wireB = DashedLine(bcol.get_right(), pivot, color=MUTED, stroke_width=2,
                           dash_length=0.09)
        feed = Arrow(pivot, probe[0].get_left(), color=CYAN, stroke_width=3.5,
                     buff=0.1, max_tip_length_to_length_ratio=0.12)
        pdot = Dot(pivot, radius=0.07, color=TEXT)
        self.play(FadeIn(acol, run_time=0.4), FadeIn(a_lbl, run_time=0.3),
                  GrowArrow(tap, run_time=0.4))
        self.play(FadeIn(sim2, run_time=0.3), GrowArrow(feedb, run_time=0.3),
                  FadeIn(bcol, run_time=0.4), FadeIn(b_lbl, run_time=0.3))
        self.play(Create(wireA), Create(wireB), FadeIn(pdot),
                  FadeIn(probe, run_time=0.4))
        # active input starts on the ENCODER (cyan)
        armA = Line(pivot, acol.get_right(), color=CYAN, stroke_width=5)
        self.play(GrowArrow(feed, run_time=0.3), Create(armA, run_time=0.3))
        metric = _chip("lift = ?", MUTED, w=1.9, h=0.85, fs=22).move_to([2.35, -0.4, 0])
        out = Arrow(probe[0].get_right(), metric[0].get_left(), color=YELLOW,
                    stroke_width=3, buff=0.12, max_tip_length_to_length_ratio=0.14)
        self.play(GrowArrow(out, run_time=0.3), FadeIn(metric, run_time=0.3))

        until(52.10)  # S9 — throw the switch to perfect physics
        self._say("We throw the switch from encoder features to perfect physics,\n"
                  "and fit the identical probe.")
        armB = Line(pivot, bcol.get_right(), color=GREEN, stroke_width=5)
        feed_g = Arrow(pivot, probe[0].get_left(), color=GREEN, stroke_width=3.5,
                       buff=0.1, max_tip_length_to_length_ratio=0.12)
        self.play(Transform(armA, armB), Transform(feed, feed_g),
                  Indicate(bcol, color=GREEN, scale_factor=1.12),
                  Flash(pivot, color=GREEN, line_length=0.18, num_lines=10, run_time=0.6))
        self.play(Indicate(probe, color=GREEN, scale_factor=1.04, run_time=0.6))

        until(58.2)   # clear during the pause before "The result…"
        self.play(FadeOut(VGroup(acol, a_lbl, tap, bcol, sim2, b_lbl, feedb,
                                 wireA, wireB, pdot, armA, feed, out, metric,
                                 probe), run_time=0.5))

        # ============ B4 — the result: perfect state still can't =============
        until(58.68)  # S10
        self._say("The result: even perfect state couldn't decode\n"
                  "the center of mass either.")
        result = VGroup(
            Text("CoM oracle lift", font_size=26, color=MUTED),
            Text("+0.0009 ± 0.0023", font_size=52, color=RED, weight="BOLD"),
            Text("→  zero  ·  the encoder null was exonerated, not passed",
                 font_size=21, color=TEXT),
        ).arrange(DOWN, buff=0.2).move_to([0, 0.2, 0])
        self.play(FadeIn(result[0], run_time=0.4))
        self.play(Write(result[1], run_time=0.7))
        self.play(FadeIn(result[2], shift=0.1 * UP, run_time=0.4))

        until(62.4)   # clear everything for the closing card
        self.play(FadeOut(result, run_time=0.4),
                  FadeOut(VGroup(blocks, clip, enc_lbl, title), run_time=0.4),
                  FadeOut(self.narr, run_time=0.3))

        # ============ closing — mirrors S11 ==================================
        until(62.74)  # S11
        closing = _fit(Text("The oracle certifies observability —\n"
                            "before we ever probe the encoder.",
                            font_size=30, color=TEXT, weight="BOLD", line_spacing=0.85), 0.92)
        closing.move_to([0, 0, 0])
        self.play(FadeIn(closing, shift=0.12 * UP, run_time=0.7))
        until(69.0)   # wav is 68.54 s — outlast it slightly so the mux never pads

    # -------- narration bar (same cross-fade convention as scene 04) --------
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
    ap.add_argument("--out", default="scene05_oracle")
    args = ap.parse_args()
    w, h, fps = _QUALITY[args.quality]
    with tempconfig({
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "background_color": BG, "output_file": args.out,
        "media_dir": "reports/video/_media", "format": "mp4",
        "disable_caching": True, "verbosity": "WARNING",
    }):
        Scene05Oracle().render()
        print("rendered:", config.output_file, f"({w}x{h}@{fps})")

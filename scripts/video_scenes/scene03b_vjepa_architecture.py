"""Scene 03b — V-JEPA 2 architecture, walked through on the *actual paper figures*.

Figure-based replacement for the schematic scene 03, now paced to the recorded
voiceover `reports/video/narration/scene03b_vjepa_architecture.wav` (66.54 s).
There are NO bottom captions — the narration carries the words, the scene only
moves a highlight region-by-region over the figures. Every beat below is cued to
the wav's silence boundaries (ffmpeg silencedetect, noise=-32dB:d=0.35); the
`until(t)` helper waits on the *absolute* scene clock so the cues stay put no
matter how long the transforms take. Curtain-raiser for the linear-probe scenes.

Two figures, cited on screen, captured WITH their printed captions at 400 dpi
(see scripts/video_scenes/README.md §"Re-capturing the paper figures"):
  * V-JEPA 2 · Assran et al., 2025 · arXiv 2506.09985 — Fig. 1, page 2 (system
    overview; the narration literally says "the second page of the paper").
    Establishes: pretrain by prediction → freeze → read with light probes. The
    "Attentive Probe Training" branch is where our experiments sit.
  * V-JEPA 2.1 · Mur-Labadia et al., 2026 · arXiv 2603.14482 — Fig. 4, page 5
    (detailed architecture). The masked-prediction *core* (x-encoder on visible
    tokens → predictor; EMA y-encoder target; L1 loss in latent space) is shared
    with V-JEPA 2.

Narrated beat order (differs from the old captioned cut: the predictor and the
V-JEPA 2.1 context-loss beats are not narrated and were dropped; the loss beat
now comes BEFORE the target encoder, following the recording):
  intro/pretraining → the figure itself → probe branch → [swap figures] →
  patchify/tubelets → mask-out → context encoder → prediction loss →
  target encoder (EMA) → closing card.

Assets live in reports/video/assets/ (extracted from papers/ via pdftoppm+crop).
    python scripts/video_scenes/scene03b_vjepa_architecture.py            # quick preview
    python scripts/video_scenes/scene03b_vjepa_architecture.py -q h       # 1080p
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from manim import (
    Scene, ImageMobject, Group, Rectangle, RoundedRectangle, Text,
    FadeIn, FadeOut, Transform, config, tempconfig,
    UP, DOWN, RIGHT,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene01_allocentric_worldlines import BG, TEXT, MUTED, CYAN, YELLOW, GREEN, BLUE

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
ASSET = os.path.join(_ROOT, "reports", "video", "assets")


def _fit(mob, frac: float = 0.9):
    if mob.width > config.frame_width * frac:
        mob.scale(config.frame_width * frac / mob.width)
    return mob


class Scene03bVJepaArchitecture(Scene):
    # geometry of the currently-shown figure (set by _show_figure)
    _imgc = np.zeros(3)
    _imgw = 1.0
    _imgh = 1.0
    _hl = None

    # ---- figure + overlay helpers -----------------------------------------
    def _show_figure(self, fname: str, height: float, center) -> Group:
        img = ImageMobject(os.path.join(ASSET, fname)).scale_to_fit_height(height)
        if img.width > config.frame_width * 0.96:          # safety for wide crops
            img.scale_to_fit_width(config.frame_width * 0.96)
        img.move_to(center)
        self._imgc = np.array(center, dtype=float)
        self._imgw, self._imgh = img.width, img.height
        border = Rectangle(width=img.width + 0.05, height=img.height + 0.05,
                           stroke_color=MUTED, stroke_width=1.5,
                           stroke_opacity=0.55).move_to(center)
        return Group(img, border)

    def _region(self, box, color, fill=0.14) -> RoundedRectangle:
        """box = (fx0, fy0, fx1, fy1) in figure fractions (x right, y down)."""
        fx0, fy0, fx1, fy1 = box
        cx, cy = self._imgc[0], self._imgc[1]
        x0 = cx + (fx0 - 0.5) * self._imgw
        x1 = cx + (fx1 - 0.5) * self._imgw
        yt = cy + (0.5 - fy0) * self._imgh
        yb = cy + (0.5 - fy1) * self._imgh
        r = RoundedRectangle(width=abs(x1 - x0), height=abs(yt - yb),
                             corner_radius=0.08, stroke_color=color, stroke_width=3.5,
                             fill_color=color, fill_opacity=fill)
        r.move_to([(x0 + x1) / 2, (yt + yb) / 2, 0])
        return r

    def _mark(self, box, color, fill=0.14):
        """Move the single highlight to a new region (fade in on first use)."""
        new = self._region(box, color, fill)
        if self._hl is None:
            self._hl = new
            self.play(FadeIn(self._hl), run_time=0.6)
        else:
            self.play(Transform(self._hl, new), run_time=0.6)

    def _clear_overlays(self, *extra):
        outs = [FadeOut(m) for m in (self._hl, *extra) if m is not None]
        if outs:
            self.play(*outs, run_time=0.6)
        self._hl = None

    # ---- the scene --------------------------------------------------------
    def construct(self):
        self.camera.background_color = BG

        def until(t: float):
            """Sleep until absolute scene time t (the wav timeline)."""
            dt = t - self.renderer.time
            if dt > 0:
                self.wait(dt)

        # ===== Part A — system overview (Fig. 1) =====
        # VO 0.0–17.9: "V-JEPA 2 is a self-supervised video model pretrained on
        # ~1M hours of internet video. It was designed by Yann LeCun…"
        titleA = _fit(Text("V-JEPA 2 — a self-supervised video model",
                           font_size=34, color=TEXT, weight="BOLD")).to_edge(UP, buff=0.32)
        self.play(FadeIn(titleA, shift=0.15 * DOWN, run_time=0.6))

        ov = self._show_figure("vjepa2_fig1_caption.png", height=6.3, center=[0, -0.28, 0])
        cite_ov = Text("V-JEPA 2 · Assran et al. 2025 · arXiv 2506.09985 — Fig. 1 (p. 2)",
                       font_size=15, color=MUTED)
        cite_ov.next_to(ov, DOWN, buff=0.08).align_to(ov, RIGHT)
        self.play(FadeIn(ov, run_time=0.8), FadeIn(cite_ov, run_time=0.5))

        until(1.3)     # "…is a self-supervised video model pretrained on 1M hours…"
        self._mark((0.030, 0.070, 0.430, 0.450), YELLOW)

        until(18.2)    # "Take a look on the Figure on the second page of the paper."
        self._mark((0.005, 0.005, 0.995, 0.710), MUTED, fill=0.04)

        until(21.9)    # "Concretely: the Attentive Probe Training section."
        self._mark((0.495, 0.215, 0.985, 0.475), CYAN)

        until(25.0)    # "That probe branch is where our experiments live." — pulse
        self.play(self._hl.animate.set_fill(opacity=0.30), run_time=0.5)
        self.play(self._hl.animate.set_fill(opacity=0.14), run_time=0.7)

        until(30.9)    # clear during the pause before "The video clip is chopped…"
        self._clear_overlays(titleA, ov, cite_ov)

        # ===== Part B — masked-prediction architecture (Fig. 4) =====
        titleB = _fit(Text("V-JEPA 2 — predicting masked video in latent space",
                           font_size=32, color=TEXT, weight="BOLD")).to_edge(UP, buff=0.32)
        arch = self._show_figure("vjepa21_fig4_caption.png", height=6.3, center=[0, -0.28, 0])
        cite_arch = Text("V-JEPA 2.1 · Mur-Labadia et al. 2026 · arXiv 2603.14482 — Fig. 4 "
                         "(core shared with V-JEPA 2)", font_size=15, color=MUTED)
        cite_arch.next_to(arch, DOWN, buff=0.08).align_to(arch, RIGHT)
        self.play(FadeIn(titleB, shift=0.15 * DOWN), FadeIn(arch), FadeIn(cite_arch),
                  run_time=0.8)

        until(32.3)    # "…chopped into 16×16-pixel, 2-frame blocks… one token each."
        self._mark((0.015, 0.010, 0.345, 0.355), YELLOW)

        until(41.5)    # "Randomly sampled patches are masked out and removed."
        self._mark((0.345, 0.012, 0.445, 0.290), YELLOW)

        until(44.3)    # "So the context encoder sees only the visible tokens."
        self._mark((0.435, 0.090, 0.600, 0.345), CYAN)

        until(47.4)    # "The loss lives in representation space… not pixels."
        self._mark((0.670, 0.530, 0.870, 0.660), YELLOW)

        until(51.9)    # "The target encoder — a slow EMA copy — full unmasked clip."
        self._mark((0.435, 0.375, 0.600, 0.690), GREEN)

        until(57.3)    # clear during the pause before the closing sentence
        self._clear_overlays(titleB, arch, cite_arch)

        # ===== closing — handoff to the probe scenes =====
        # VO 57.9–65.3: "We freeze all 40 transformer blocks and read each one
        # with a linear probe: which layers expose physics?"
        closing = _fit(Text("We freeze all 40 transformer blocks and read each one\n"
                            "with a linear probe — which layers expose physics?",
                            font_size=30, color=TEXT, weight="BOLD", line_spacing=0.8), 0.9)
        closing.move_to([0, 0.2, 0])
        self.play(FadeIn(closing, shift=0.12 * UP, run_time=0.8))
        until(66.7)    # wav is 66.54 s — outlast it slightly so the mux never pads


# --------------------------------- rendering --------------------------------
_QUALITY = {"l": (854, 480, 15), "m": (1280, 720, 30), "h": (1920, 1080, 30)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", choices=list(_QUALITY), default="l")
    ap.add_argument("--out", default="scene03b_vjepa_architecture")
    args = ap.parse_args()
    w, h, fps = _QUALITY[args.quality]
    with tempconfig({
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "background_color": BG, "output_file": args.out,
        "media_dir": "reports/video/_media", "format": "mp4",
        "disable_caching": True, "verbosity": "WARNING",
    }):
        Scene03bVJepaArchitecture().render()
        print("rendered:", config.output_file, f"({w}x{h}@{fps})")

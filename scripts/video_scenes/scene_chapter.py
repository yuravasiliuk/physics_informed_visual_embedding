"""Scene — chapter transition card (~5 s). Announces the upcoming chapter/scene.

These cards give the deck a "chapters" structure and a breath between content
scenes (better perception of the material). One card is rendered per chapter via
--index/--title/--subtitle; `finalize_deck.py` assembles them *before* each
content scene and swells the music bed to 2x baseline across every card, ducking
back to baseline as the next narrated scene starts.

Palette + fonts are reused from scene01 so the cards match the deck. Self-
contained and cwd-robust like the sibling scenes; the card is a fixed ~5 s
animation (fade the kicker/title/rule/subtitle in, hold, fade out) — it is NOT
cued to a wav (cards carry no narration).

    python scripts/video_scenes/scene_chapter.py --index 4 \
        --title "What is a linear probe?" \
        --subtitle "Reading physics out of frozen features" -q h --out chapter4
"""
from __future__ import annotations

import argparse
import os
import sys
from manim import (
    Scene, VGroup, Text, Line, FadeIn, FadeOut, Create,
    config, tempconfig, UP, DOWN, LEFT, RIGHT,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene01_allocentric_worldlines import BG, TEXT, MUTED, CYAN


def _fit(mob, frac: float = 0.86):
    if mob.width > config.frame_width * frac:
        mob.scale(config.frame_width * frac / mob.width)
    return mob


class ChapterCard(Scene):
    # set by __main__ before render()
    index = 1
    title_text = "Chapter title"
    subtitle_text = ""

    def construct(self):
        self.camera.background_color = BG

        # "CHAPTER N" kicker — spaced caps for a title-card feel
        kicker = Text(f"C H A P T E R   {self.index}", font_size=24,
                      color=CYAN, weight="BOLD").move_to([0, 1.25, 0])
        title = _fit(Text(self.title_text, font_size=54, color=TEXT,
                          weight="BOLD")).move_to([0, 0.32, 0])
        # accent rule under the title, half the title's width, grows in
        half = max(1.6, min(title.width, config.frame_width * 0.8) * 0.5)
        rule = Line([-half, -0.35, 0], [half, -0.35, 0],
                    stroke_color=CYAN, stroke_width=3)
        subtitle = _fit(Text(self.subtitle_text, font_size=27, color=MUTED),
                        0.8).move_to([0, -0.95, 0]) if self.subtitle_text else None

        self.play(FadeIn(kicker, shift=0.2 * DOWN), run_time=0.5)
        self.play(FadeIn(title, shift=0.16 * UP), run_time=0.7)
        self.play(Create(rule), run_time=0.5)
        parts = [kicker, title, rule]
        if subtitle is not None:
            self.play(FadeIn(subtitle, shift=0.1 * UP), run_time=0.5)
            parts.append(subtitle)
        else:
            self.wait(0.5)
        self.wait(2.1)                       # hold the card
        self.play(FadeOut(VGroup(*parts)), run_time=0.7)   # dissolve into the scene


# --------------------------------- rendering --------------------------------
_QUALITY = {"l": (854, 480, 15), "m": (1280, 720, 30), "h": (1920, 1080, 30)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", choices=list(_QUALITY), default="l")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    w, h, fps = _QUALITY[args.quality]
    ChapterCard.index = args.index
    ChapterCard.title_text = args.title
    ChapterCard.subtitle_text = args.subtitle
    with tempconfig({
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "background_color": BG, "output_file": args.out or f"chapter{args.index}",
        "media_dir": "reports/video/_media", "format": "mp4",
        "disable_caching": True, "verbosity": "WARNING",
    }):
        ChapterCard().render()
        print("rendered:", config.output_file, f"({w}x{h}@{fps})")

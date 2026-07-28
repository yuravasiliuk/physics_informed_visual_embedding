"""Finalize the deck: chapter cards + background music (swelled over the cards) + logo.

Takes the already-muxed 1080p scene chunks (from build_deck.py) and the rendered
chapter cards (scene_chapter.py), and produces the shipped final in two ffmpeg
passes plus a numpy step:

  1. ASSEMBLE (one video encode): concat  card1 · scene1 · card2 · scene2 · … ·
     card5 · scene5  (chapter card BEFORE each content scene, no gaps — each card
     fades itself in/out), normalizing resolution/fps, and burn the logo into the
     bottom-left corner. Narration audio is carried through; cards are silent.
     → <result_dir>/<chapters name> (logo burned, narration only).

  2. MUSIC BED (numpy): tile `music.wav` to the deck length with crossfaded seams
     (a hard loop pumps — the clip fades in and ends loud), then multiply it by a
     volume ENVELOPE:
         baseline  = --music_volume            (during narrated scenes)
         over each chapter card the bed swells to  --music_volume × --transition_boost
         (ramp up --ramp_up s at the card start, hold, then ramp DOWN --ramp_down s
         back to baseline once the next narrated scene starts)
     plus a global fade in/out. The envelope is baked into the bed samples, so the
     mix needs no ffmpeg volume expressions.

  3. MIX (video copied — no second video encode): amix the enveloped bed under the
     narration (normalize=0 keeps the voice at full level; alimiter guards clips).
     → <result_dir>/<name>   (default deck_1080p_final.mp4)

Reuses build_deck.probe / BG_HEX (read-only import — build_deck is untouched).
Run from the repo root; needs ffmpeg/ffprobe + numpy. No manim.

    # defaults: 5 scene chunks in reports/video/1080p, 5 cards in _media/1080p30
    python3 scripts/video_scenes/finalize_deck.py

    # louder bed, bigger swell, smaller logo
    python3 scripts/video_scenes/finalize_deck.py \
        --music_volume 0.28 --transition_boost 2.0 --logo_scale 120
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_deck import probe, BG_HEX  # read-only reuse; build_deck stays untouched

SR = 48000

# deck order — a chapter card is rendered/placed BEFORE each content scene
CARDS = ["chapter1", "chapter2", "chapter3", "chapter4", "chapter5"]
SCENES = [
    "scene01_exocentric_worldlines",
    "scene02_egocentric_representation",
    "scene03b_vjepa_architecture",
    "scene04_linear_probe",
    "scene05_oracle",
]


# ------------------------------- music bed ----------------------------------
def build_bed(music: Path, target_len: float, out: Path, xfade: float = 0.4) -> None:
    """Tile `music` to >= target_len with `xfade`-second crossfaded seams
    (no click, no pump), as a mono 48 kHz pcm wav."""
    clip = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(music)]))
    # len(M) = clip + (M-1)*(clip - xfade)  →  smallest M covering target (+1 margin)
    M = int(np.ceil((target_len - clip) / (clip - xfade))) + 2
    cmd = ["ffmpeg", "-y", "-v", "error"] + ["-i", str(music)] * M
    g = [f"[0:a][1:a]acrossfade=d={xfade}:c1=tri:c2=tri[x1]"]
    g += [f"[x{i-1}][{i}:a]acrossfade=d={xfade}:c1=tri:c2=tri[x{i}]"
          for i in range(2, M)]
    subprocess.run(cmd + ["-filter_complex", ";".join(g), "-map", f"[x{M-1}]",
                          "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
                          str(out)], check=True)


def envelope(total: float, card_windows: list[tuple[float, float]], *,
             base: float, boost: float, ru: float, rd: float,
             fade_in: float, fade_out: float) -> np.ndarray:
    """Per-sample gain over the deck timeline: baseline `base`, swelling to
    base*boost across each card (ramp up `ru` at card start, ramp down `rd` into
    the following scene), times a global fade in/out."""
    n = int(round(total * SR))
    t = np.arange(n) / SR
    b = np.ones(n)                                   # boost factor, 1 .. boost
    for cs, ce in card_windows:
        up = (t >= cs) & (t < cs + ru)
        b[up] = 1 + (boost - 1) * (t[up] - cs) / ru
        hold = (t >= cs + ru) & (t < ce)
        b[hold] = boost
        dn = (t >= ce) & (t < ce + rd)               # ramp down INTO the scene
        b[dn] = boost - (boost - 1) * (t[dn] - ce) / rd
    g = base * b
    if fade_in > 0:
        fi = t < fade_in
        g[fi] *= t[fi] / fade_in
    if fade_out > 0:
        fo = t > total - fade_out
        g[fo] *= np.clip((total - t[fo]) / fade_out, 0, 1)
    return g


def apply_envelope(bed: Path, env: np.ndarray, out: Path) -> None:
    w = wave.open(str(bed), "rb")
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if w.getnchannels() > 1:
        x = x.reshape(-1, w.getnchannels()).mean(1)
    x = x[:len(env)] / 32768.0
    if len(x) < len(env):                            # bed shorter than deck (shouldn't happen)
        x = np.pad(x, (0, len(env) - len(x)))
    y = np.clip(x * env, -1.0, 1.0)
    ww = wave.open(str(out), "wb")
    ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(SR)
    ww.writeframes((y * 32767).astype(np.int16).tobytes())
    ww.close()


# ------------------------------- assembly -----------------------------------
def assemble(segs: list[Path], metas: list[dict], logo: Path, out: Path, *,
             logo_scale: int, logo_margin: int, logo_opacity: float, crf: int) -> None:
    """Concat the segments (normalized), carry narration / silence, burn the logo."""
    W = max(m["w"] for m in metas); H = max(m["h"] for m in metas)
    FPS = max(m["fps"] for m in metas)
    AFMT = "aformat=sample_fmts=fltp:channel_layouts=mono"
    parts, labels = [], []
    for i, m in enumerate(metas):
        parts.append(f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                     f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
                     f"fps={FPS:g},setsar=1,format=yuv420p[v{i}]")
        if m["audio"]:
            parts.append(f"[{i}:a]aresample={SR},{AFMT},apad,"
                         f"atrim=0:{m['dur']:.3f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            parts.append(f"anullsrc=r={SR}:cl=mono,{AFMT},"
                         f"atrim=0:{m['dur']:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    parts.append("".join(labels) + f"concat=n={len(metas)}:v=1:a=1[cv][ca]")
    parts.append(f"[{len(metas)}:v]scale={logo_scale}:-1,format=rgba,"
                 f"colorchannelmixer=aa={logo_opacity}[lg]")
    parts.append(f"[cv][lg]overlay={logo_margin}:main_h-overlay_h-{logo_margin}[outv]")
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for s in segs:
        cmd += ["-i", str(s)]
    cmd += ["-i", str(logo), "-filter_complex", ";".join(parts),
            "-map", "[outv]", "-map", "[ca]",
            "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene_dir", default="reports/video/1080p",
                    help="dir with the muxed scene chunks")
    ap.add_argument("--card_dir", default="reports/video/_media/videos/1080p30",
                    help="dir with the rendered chapter cards")
    ap.add_argument("--music", default="reports/video/assets/music.wav")
    ap.add_argument("--logo", default="reports/video/assets/logo_gl.png")
    ap.add_argument("--result_dir", default="reports/video/1080p")
    ap.add_argument("--name", default="deck_1080p_final.mp4")
    ap.add_argument("--chapters_name", default="deck_1080p_chapters.mp4",
                    help="intermediate: chaptered deck, logo burned, narration only")
    # ---- the levers -------------------------------------------------------
    ap.add_argument("--music_volume", type=float, default=0.23,
                    help="baseline bed gain during narrated scenes")
    ap.add_argument("--transition_boost", type=float, default=2.0,
                    help="bed multiplier over each chapter card")
    ap.add_argument("--ramp_up", type=float, default=0.8,
                    help="s to swell to the boosted level at a card's start")
    ap.add_argument("--ramp_down", type=float, default=1.5,
                    help="s to duck back to baseline once the next scene starts")
    ap.add_argument("--fade_in", type=float, default=1.5)
    ap.add_argument("--fade_out", type=float, default=3.0)
    ap.add_argument("--logo_scale", type=int, default=150, help="logo width in px")
    ap.add_argument("--logo_margin", type=int, default=44, help="logo inset in px")
    ap.add_argument("--logo_opacity", type=float, default=0.9)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--xfade", type=float, default=0.4, help="music loop crossfade")
    args = ap.parse_args()

    scene_dir, card_dir = Path(args.scene_dir), Path(args.card_dir)
    result_dir = Path(args.result_dir); result_dir.mkdir(parents=True, exist_ok=True)

    # interleave: card1, scene1, card2, scene2, ...  (card before each scene)
    segs: list[Path] = []
    for c, s in zip(CARDS, SCENES):
        segs += [card_dir / f"{c}.mp4", scene_dir / f"{s}.mp4"]
    missing = [str(p) for p in segs if not p.exists()]
    if missing:
        raise SystemExit("missing segments:\n  " + "\n  ".join(missing))

    metas = [probe(p) for p in segs]
    # cumulative boundaries; the card segments (even indices) are the swell windows
    starts, acc = [], 0.0
    for m in metas:
        starts.append(acc); acc += m["dur"]
    total = acc
    card_windows = [(starts[i], starts[i] + metas[i]["dur"])
                    for i in range(0, len(segs), 2)]           # cards at even indices

    print(f"segments: {len(segs)}  total: {total:.2f}s  "
          f"cards at: {', '.join(f'{a:.1f}-{b:.1f}' for a, b in card_windows)}")

    # 1) assemble chaptered deck + logo (one video encode)
    chapters = result_dir / args.chapters_name
    assemble(segs, metas, Path(args.logo), chapters,
             logo_scale=args.logo_scale, logo_margin=args.logo_margin,
             logo_opacity=args.logo_opacity, crf=args.crf)
    print(f"assembled {chapters}")

    # 2) music bed → enveloped bed (numpy)
    raw_bed = result_dir / "_music_bed_raw.wav"
    env_bed = result_dir / "_music_bed_env.wav"
    build_bed(Path(args.music), total + 2.0, raw_bed, xfade=args.xfade)
    env = envelope(total, card_windows, base=args.music_volume,
                   boost=args.transition_boost, ru=args.ramp_up, rd=args.ramp_down,
                   fade_in=args.fade_in, fade_out=args.fade_out)
    apply_envelope(raw_bed, env, env_bed)
    print(f"bed: baseline {args.music_volume} → swell {args.music_volume*args.transition_boost:.2f} on cards")

    # 3) mix bed under narration; copy the (logo-burned) video → final
    final = result_dir / args.name
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(chapters), "-i", str(env_bed),
         "-filter_complex",
         "[0:a][1:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.98[aout]",
         "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)],
        check=True)
    raw_bed.unlink(missing_ok=True); env_bed.unlink(missing_ok=True)
    d = probe(final)
    print(f"final {final}  ({d['w']}x{d['h']}@{d['fps']:g}, {d['dur']:.2f}s)")


if __name__ == "__main__":
    main()

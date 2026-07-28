"""Build the presentation deck video: mux narration + concatenate the chunks.

Deck order (reports/video/NARRATION.md): 01 → 02 → 03b → 04 → 05.

Two jobs, in one pass over the deck:
  1. MUX — for any scene whose mp4 in --src has NO audio track but has a
     matching wav in --narration_dir (matched on the sceneNN_ prefix, so
     scene01's wav keeps its historical "allocentric" filename), attach the
     narration; if the wav outruns the video, the last frame is frozen
     (tpad clone) to cover it. Muxed per-scene files land in --result_dir.
  2. CONCAT — join the five chunks with a background-colour gap between
     scenes. Every segment is normalized to the largest resolution / highest
     fps in the set (scale + BG-colour pad + fps + yuv420p), and every
     segment's audio is forced to exactly its video duration (silence for
     the unnarrated scenes, apad+atrim otherwise) so A/V sync can't drift
     across the joins.

Needs only ffmpeg/ffprobe on PATH — no manim. Run from the repo root.

    # canonical 480p deck from the already-muxed chunks in reports/video/
    python3 scripts/video_scenes/build_deck.py

    # 1080p: render the scenes first (-q h), then build from the raw renders
    python3 scripts/video_scenes/build_deck.py \
        --src reports/video/_media/videos/1080p30 --result_dir reports/video/1080p

    # only some scenes (any order, ".mp4" optional; default is the full deck)
    python3 scripts/video_scenes/build_deck.py \
        --src reports/video/_media/videos/1080p30 --result_dir reports/video/1080p \
        --scenes scene01_exocentric_worldlines scene02_egocentric_representation
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

BG_HEX = "0x0B1020"  # scene background (BG in scene01's palette) — pads & gaps

DECK = [
    "scene01_exocentric_worldlines",
    "scene02_egocentric_representation",
    "scene03b_vjepa_architecture",
    "scene04_linear_probe",
    "scene05_oracle",
]


def probe(path: Path) -> dict:
    """Width/height/fps/duration + whether the file has an audio stream."""
    meta = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)]))
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {
        "w": int(v["width"]), "h": int(v["height"]), "fps": num / den,
        "dur": float(v.get("duration") or meta["format"]["duration"]),
        "audio": any(s["codec_type"] == "audio" for s in meta["streams"]),
    }


def wav_duration(path: Path) -> float:
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)]))


def find_wav(narration_dir: Path, stem: str) -> Path | None:
    """Match narration by the sceneNN_ prefix (wav basenames may lag renames)."""
    hits = sorted(narration_dir.glob(f"{stem.split('_')[0]}_*.wav"))
    return hits[0] if hits else None


def mux(video: Path, wav: Path, out: Path, crf: int) -> None:
    """Attach the wav; freeze the last frame if the audio is longer."""
    if out.resolve() == video.resolve():
        raise SystemExit(f"mux would overwrite its own input: {video} — "
                         f"pass a --result_dir different from --src")
    pad = max(0.0, wav_duration(wav) - probe(video)["dur"])
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(wav),
         "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[v]",
         "-map", "[v]", "-map", "1:a",
         "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", str(out)], check=True)
    print(f"muxed  {out.name}: +{pad:.2f}s last-frame freeze ({wav.name})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="reports/video",
                    help="dir with the five scene mp4s (raw renders or muxed chunks)")
    ap.add_argument("--narration_dir", default="reports/video/narration")
    ap.add_argument("--result_dir", default="reports/video",
                    help="where muxed chunks and the deck mp4 are written")
    ap.add_argument("--name", default=None,
                    help="deck filename (default: deck_<height>p.mp4)")
    ap.add_argument("--scenes", nargs="+", default=DECK, metavar="SCENE",
                    help="scene mp4 basenames to join, in this order "
                         "(default: the full deck)")
    ap.add_argument("--gap", type=float, default=0.5,
                    help="seconds of background-colour pause between scenes")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    src, ndir = Path(args.src), Path(args.narration_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # accept bare stems, "name.mp4", or paths — only the basename is used
    deck = [Path(s).name.removesuffix(".mp4") for s in args.scenes]

    missing = [s for s in deck if not (src / f"{s}.mp4").exists()]
    if missing:
        raise SystemExit(f"missing in {src}: {', '.join(missing)}")

    # --- 1. mux narration where a scene is silent but a wav exists ----------
    clips: list[Path] = []
    for stem in deck:
        clip = src / f"{stem}.mp4"
        wav = find_wav(ndir, stem)
        if wav is not None and not probe(clip)["audio"]:
            muxed = result_dir / f"{stem}.mp4"
            mux(clip, wav, muxed, args.crf)
            clip = muxed
        clips.append(clip)

    # --- 2. concat with normalization + inter-scene gaps --------------------
    metas = [probe(c) for c in clips]
    W = max(m["w"] for m in metas)
    H = max(m["h"] for m in metas)
    FPS = max(m["fps"] for m in metas)
    AFMT = "aformat=sample_fmts=fltp:channel_layouts=mono"

    parts: list[str] = []
    labels: list[str] = []
    for i, m in enumerate(metas):
        parts.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
            f"fps={FPS:g},setsar=1,format=yuv420p[v{i}]")
        if m["audio"]:  # pin audio to the exact video duration (no drift)
            parts.append(f"[{i}:a]aresample=48000,{AFMT},apad,"
                         f"atrim=0:{m['dur']:.3f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=mono,{AFMT},"
                         f"atrim=0:{m['dur']:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[v{i}][a{i}]")
        if args.gap > 0 and i < len(metas) - 1:
            parts.append(f"color=c={BG_HEX}:s={W}x{H}:r={FPS:g}:d={args.gap:g},"
                         f"setsar=1,format=yuv420p[gv{i}]")
            parts.append(f"anullsrc=r=48000:cl=mono,{AFMT},"
                         f"atrim=0:{args.gap:g}[ga{i}]")
            labels.append(f"[gv{i}][ga{i}]")

    parts.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=1[outv][outa]")

    out = result_dir / (args.name or f"deck_{H}p.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for c in clips:
        cmd += ["-i", str(c)]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)

    d = probe(out)
    total = sum(m["dur"] for m in metas) + args.gap * (len(metas) - 1)
    print(f"deck   {out}  ({d['w']}x{d['h']}@{d['fps']:g}, "
          f"{d['dur']:.2f}s; expected ≈{total:.2f}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build narrated videos from matching PPTX + slide-script TXT files."""

from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SLIDE_HEADER = re.compile(r"^SLIDE\s+(\d+)\s*$", re.IGNORECASE)
SEPARATOR = re.compile(r"^=+$")
FOOTER = re.compile(
    r"^(VEB Healthcare Private Ltd.*|"
    r"Week\s+\d+\s+[•·].*|"
    r"\d+\s*/\s*\d+)\s*$",
    re.IGNORECASE,
)
AUDIO_EXTS = {".wav", ".mp3", ".m4a"}
WEEK_DAY_STEM = re.compile(r"^Week(\d+)_Day(\d+)", re.IGNORECASE)
PER_SLIDE_NAME = re.compile(r"^W(\d+)D(\d+)_slide(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pptx-root",
        type=Path,
        default=Path(r"C:\Users\MM\Downloads\RCM Training Modules"),
    )
    parser.add_argument(
        "--txt-root",
        type=Path,
        default=Path(r"C:\Users\MM\Downloads\RCM_Training_Modules_TXT"),
        help="Combined SLIDE N .txt files (legacy)",
    )
    parser.add_argument(
        "--scripts-root",
        type=Path,
        default=Path(r"C:\Users\MM\Downloads\Narration Scripts\Narration Scripts"),
        help="Per-slide narration folder (Day1/W1D1_slide01.txt, ...)",
    )
    parser.add_argument("--out", type=Path, default=root / "RCM_Videos")
    parser.add_argument(
        "--tts-engine",
        choices=("kokoro", "gemini"),
        default="kokoro",
        help="Local Kokoro ONNX, or Gemini cloud TTS (needs GEMINI_API_KEY)",
    )
    parser.add_argument("--model", type=Path, default=root / "kokoro-v1.0.onnx")
    parser.add_argument("--voices", type=Path, default=root / "voices-v1.0.bin")
    parser.add_argument(
        "--voice",
        default="af_bella",
        help="Kokoro voice name, or blend like af_bella:70,af_sarah:30",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-3.1-flash-tts-preview",
        help="Gemini TTS model id",
    )
    parser.add_argument(
        "--gemini-voice",
        default="Kore",
        help="Gemini prebuilt voice (e.g. Kore, Aoede, Charon, Fenrir, Puck)",
    )
    parser.add_argument("--lang", default="en-us")
    parser.add_argument("--speed", type=float, default=0.88)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--only", help="Process modules whose stem contains this text")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N modules")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    return parser.parse_args()


def find_ffmpeg(explicit: str) -> str:
    if Path(explicit).is_file():
        return explicit
    found = shutil.which(explicit) or shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg not found on PATH. Install Gyan.FFmpeg or pass --ffmpeg.")


def parse_slide_scripts(txt_path: Path) -> dict[int, str]:
    slides: dict[int, list[str]] = {}
    current: int | None = None
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        match = SLIDE_HEADER.match(line)
        if match:
            current = int(match.group(1))
            slides[current] = []
            continue
        if current is None:
            continue
        if not line or SEPARATOR.match(line) or FOOTER.match(line):
            continue
        slides[current].append(line)

    cleaned: dict[int, str] = {}
    for number, lines in slides.items():
        parts: list[str] = []
        for line in lines:
            line = re.sub(r"^[•\-–]\s*", "", line).strip()
            if not line:
                continue
            if line[-1] not in ".!?:;":
                line += "."
            parts.append(line)
        text = " ".join(parts).strip()
        if text:
            cleaned[number] = text
    return cleaned


def week_folder_for(pptx_path: Path) -> str:
    match = WEEK_DAY_STEM.match(pptx_path.stem)
    if match:
        return f"Week{int(match.group(1))}"
    parent = pptx_path.parent.name
    if re.match(r"^Week\d+$", parent, re.IGNORECASE):
        return parent
    return parent or "misc"


def load_per_slide_scripts(scripts_root: Path, week: int, day: int) -> dict[int, str]:
    """Load DayN/W{week}D{day}_slideNN.txt narration files."""
    if not scripts_root or not scripts_root.exists():
        return {}
    cleaned: dict[int, str] = {}
    for path in scripts_root.rglob("*.txt"):
        stem = path.stem
        if "REVIEW" in stem.upper() or "ALL_SLIDES" in stem.upper():
            continue
        match = PER_SLIDE_NAME.match(stem)
        if not match:
            continue
        if int(match.group(1)) != week or int(match.group(2)) != day:
            continue
        text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).strip()
        if text:
            cleaned[int(match.group(3))] = text
    return cleaned


def load_scripts_for_pptx(
    pptx_path: Path,
    scripts_root: Path | None,
    txt_root: Path | None,
) -> dict[int, str]:
    match = WEEK_DAY_STEM.match(pptx_path.stem)
    if match and scripts_root:
        scripts = load_per_slide_scripts(
            scripts_root, int(match.group(1)), int(match.group(2))
        )
        if scripts:
            return scripts
    if txt_root and txt_root.exists():
        for txt_path in txt_root.rglob(f"{pptx_path.stem}.txt"):
            return parse_slide_scripts(txt_path)
    return {}


def pair_modules(
    pptx_root: Path,
    txt_root: Path | None,
    scripts_root: Path | None,
) -> list[tuple[Path, dict[int, str]]]:
    pptx_files = sorted({p.resolve(): p for p in pptx_root.rglob("*.pptx")}.values(), key=lambda p: p.stem)
    pairs: list[tuple[Path, dict[int, str]]] = []
    missing: list[str] = []
    for pptx_path in pptx_files:
        scripts = load_scripts_for_pptx(pptx_path, scripts_root, txt_root)
        if scripts:
            pairs.append((pptx_path, scripts))
        else:
            missing.append(pptx_path.stem)
    if missing:
        print(f"Warning: PPTX without narration scripts ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
    return pairs


def export_slides_png(pptx_path: Path, slides_dir: Path, width: int, height: int) -> list[Path]:
    import pythoncom
    import win32com.client

    slides_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(slides_dir.glob("slide-*.png"))
    if existing:
        return existing

    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx("PowerPoint.Application")
    try:
        app.Visible = True
    except Exception:
        pass

    presentation = app.Presentations.Open(
        str(pptx_path.resolve()),
        ReadOnly=True,
        Untitled=False,
        WithWindow=False,
    )
    try:
        for index in range(1, presentation.Slides.Count + 1):
            dest = slides_dir / f"slide-{index:02d}.png"
            presentation.Slides(index).Export(str(dest), "PNG", width, height)
    finally:
        presentation.Close()
        app.Quit()
        pythoncom.CoUninitialize()

    images = sorted(slides_dir.glob("slide-*.png"))
    if not images:
        raise RuntimeError(f"PowerPoint exported no PNG slides from {pptx_path}")
    return images


def prepare_narration(text: str) -> str:
    """Make slide text speak more naturally for TTS."""
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        r"\bRCM\b": "R C M",
        r"\bKPI\b": "K P I",
        r"\bKPIs\b": "K P I's",
        r"\bROI\b": "R O I",
        r"\bCFO\b": "C F O",
        r"\bCIO\b": "C I O",
        r"\bCRM\b": "C R M",
        r"\bEHR\b": "E H R",
        r"\bIDN\b": "I D N",
        r"\bIDNs\b": "I D N's",
        r"\bAMC\b": "A M C",
        r"\bAMCs\b": "A M C's",
        r"\bASC\b": "A S C",
        r"\bASCs\b": "A S C's",
        r"\bFQHC\b": "F Q H C",
        r"\bFQHCs\b": "F Q H C's",
        r"\bHFMA\b": "H F M A",
        r"\bAAPC\b": "A A P C",
        r"\bAHIMA\b": "A H I M A",
        r"\bLMS\b": "L M S",
        r"\bUSA\b": "U S A",
        r"\bUS\b": "U S",
        r"\bB2B\b": "B to B",
        r"≥\s*": "at least ",
        r"≤\s*": "at most ",
        r"–|—": ", ",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def normalize_audio(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-6:
        return audio
    audio = audio * (peak / max_abs)
    # Gentle fade in/out to avoid clicks
    fade = min(int(0.02 * 24000), max(1, audio.size // 20))
    if audio.size > fade * 2:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return np.clip(audio, -1.0, 1.0)


def synthesize_slide_kokoro(
    kokoro: Any,
    text: str,
    wav_path: Path,
    voice: str | np.ndarray,
    speed: float,
    lang: str,
) -> None:
    from kokoro_tts import chunk_text, process_chunk_sequential

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    text = prepare_narration(text)
    # Prefer fewer, larger chunks so speech stays continuous and less robotic.
    chunks = chunk_text(text, initial_chunk_size=1800)
    all_samples: list[np.ndarray] = []
    sample_rate = 24000
    pause = np.zeros(int(0.22 * sample_rate), dtype=np.float32)
    for index, chunk in enumerate(chunks):
        samples, rate = process_chunk_sequential(chunk, kokoro, voice, speed, lang)
        if samples is None or rate is None:
            raise RuntimeError(f"TTS failed for chunk: {chunk[:80]!r}")
        sample_rate = rate
        piece = np.asarray(samples, dtype=np.float32)
        all_samples.append(piece)
        if index < len(chunks) - 1:
            all_samples.append(pause)
    if not all_samples:
        audio = np.zeros(int(sample_rate), dtype=np.float32)
    else:
        lead = np.zeros(int(0.25 * sample_rate), dtype=np.float32)
        trail = np.zeros(int(0.45 * sample_rate), dtype=np.float32)
        audio = np.concatenate([lead, *all_samples, trail])
    audio = normalize_audio(audio)
    sf.write(str(wav_path), audio, sample_rate)


def write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def synthesize_slide_gemini(
    client: Any,
    text: str,
    wav_path: Path,
    model: str,
    voice_name: str,
    retries: int = 6,
) -> None:
    from google.genai import types

    text = prepare_narration(text)
    # Controllable TTS: keep delivery clear and training-friendly.
    prompt = (
        "Speak in a clear, professional training-narrator style at a steady pace. "
        "Do not add extra words. Recite exactly:\n\n"
        f"{text}"
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                ),
            )
            try:
                raw = response.candidates[0].content.parts[0].inline_data.data
            except (AttributeError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Gemini TTS returned no audio: {response}") from exc
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            write_pcm16_wav(wav_path, raw, sample_rate=24000)
            return
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = min(60.0, 5.0 * (2**attempt))
                # Honor retryDelay=Ns if present
                delay_match = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
                if delay_match:
                    wait = max(wait, float(delay_match.group(1)) + 1.0)
                print(f"    rate-limited; sleep {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Gemini TTS failed after {retries} retries: {last_exc}")


def make_slide_video(ffmpeg: str, image: Path, audio: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loop", "1",
        "-framerate", "30",
        "-i", str(image),
        "-i", str(audio),
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def concat_videos(ffmpeg: str, videos: list[Path], concat_list: Path, final_video: Path) -> None:
    concat_list.parent.mkdir(parents=True, exist_ok=True)
    final_video.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for video in videos:
        path = str(video.resolve()).replace("'", "'\\''")
        lines.append(f"file '{path}'")
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copy_cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list), "-c", "copy", str(final_video),
    ]
    result = subprocess.run(copy_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return
    reencode_cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(final_video),
    ]
    subprocess.run(reencode_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def process_module(
    pptx_path: Path,
    scripts: dict[int, str],
    out_root: Path,
    tts: Any,
    ffmpeg: str,
    args: argparse.Namespace,
) -> Path:
    stem = pptx_path.stem
    week = week_folder_for(pptx_path)
    work = out_root / "_work" / week / stem
    slides_dir = work / "slides"
    audio_dir = work / "audio"
    videos_dir = work / "slide_videos"
    final_dir = out_root / week
    final_video = final_dir / f"{stem}.mp4"

    if final_video.exists() and not args.force:
        print(f"SKIP (exists): {final_video}")
        return final_video

    print(f"\n=== {week}/{stem} ===")
    print(f"  Narration slides: {len(scripts)}")
    images = export_slides_png(pptx_path, slides_dir, args.width, args.height)
    count = min(len(images), max(scripts) if scripts else 0)
    if count == 0:
        raise RuntimeError(f"No slides/scripts found for {stem}")
    if len(images) != len(scripts):
        print(
            f"Warning: {len(images)} PPTX slides vs {len(scripts)} script slides; "
            f"using {count}."
        )

    slide_videos: list[Path] = []
    for index in range(1, count + 1):
        image = images[index - 1]
        audio = audio_dir / f"{index}.wav"
        clip = videos_dir / f"slide_{index:02d}.mp4"
        text = scripts.get(index)
        if not text:
            raise RuntimeError(f"{stem}: missing script for slide {index}")

        if args.force or not audio.exists():
            print(f"  TTS ({args.tts_engine}) slide {index}/{count}...")
            if args.tts_engine == "gemini":
                synthesize_slide_gemini(
                    tts, text, audio, args.gemini_model, args.gemini_voice
                )
            else:
                synthesize_slide_kokoro(
                    tts, text, audio, args.voice, args.speed, args.lang
                )
        else:
            print(f"  Reuse audio slide {index}/{count}")

        if args.force or not clip.exists():
            print(f"  Encode slide {index}/{count}...")
            make_slide_video(ffmpeg, image, audio, clip)
        else:
            print(f"  Reuse video slide {index}/{count}")
        slide_videos.append(clip)

    print("  Combining slides...")
    concat_videos(ffmpeg, slide_videos, work / "concat.txt", final_video)
    print(f"  Wrote {final_video}")
    return final_video


def main() -> int:
    args = parse_args()
    ffmpeg = find_ffmpeg(args.ffmpeg)

    pairs = pair_modules(args.pptx_root, args.txt_root, args.scripts_root)
    if args.only:
        needle = args.only.lower()
        pairs = [p for p in pairs if needle in p[0].stem.lower() or needle in str(p[0]).lower()]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("No matching PPTX + narration script modules found.")
        return 1

    print(f"Modules to process: {len(pairs)}")
    print(f"Scripts root: {args.scripts_root}")
    print(f"TTS engine: {args.tts_engine}")

    tts: Any
    if args.tts_engine == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for --tts-engine gemini.")
            return 1
        from google import genai

        tts = genai.Client(api_key=api_key)
        print(f"Gemini model: {args.gemini_model} | voice: {args.gemini_voice}")
    else:
        if not args.model.exists() or not args.voices.exists():
            print("Missing Kokoro model files. Expected:")
            print(f"  {args.model}")
            print(f"  {args.voices}")
            return 1
        from kokoro_onnx import Kokoro
        from kokoro_tts import validate_voice

        print("Loading Kokoro model...")
        tts = Kokoro(str(args.model), str(args.voices))
        try:
            voice = validate_voice(args.voice, tts)
        except Exception as exc:
            print(f"Invalid voice '{args.voice}': {exc}")
            return 1
        args.voice = voice
        print(
            f"Voice: {args.voice if isinstance(args.voice, str) else 'blend'} "
            f"@ speed {args.speed}"
        )

    failures: list[str] = []
    started = time.time()
    for pptx_path, scripts in pairs:
        try:
            process_module(pptx_path, scripts, args.out, tts, ffmpeg, args)
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            print(f"FAILED {pptx_path.stem}: ffmpeg error\n{err[-2000:]}")
            failures.append(pptx_path.stem)
        except Exception as exc:
            print(f"FAILED {pptx_path.stem}: {exc}")
            failures.append(pptx_path.stem)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed/60:.1f} min. Failures: {len(failures)}")
    for name in failures:
        print(f"  {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

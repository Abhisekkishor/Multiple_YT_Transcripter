#!/usr/bin/env python
"""
Batch‑download YouTube audio and transcribe with faster‑whisper (GPU first, CPU fallback).

Usage (CLI):
    python multiple_youtube_transcripter.py <url_or_id> <url_or_id> ...

If running in Colab the One‑Cell launcher (see README) handles everything.
"""
import os, sys, re, shutil, yt_dlp, torch
from pathlib import Path
from faster_whisper import WhisperModel

# ──────────────────────────────────────────────────────────────────────────────
MODEL_SIZE   = "tiny"            # tiny = fastest; switch to "small" if needed
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
HOME_DRIVE   = Path("/content/drive/MyDrive")        # Colab drive mount
SAVE_DIR     = (HOME_DRIVE / "YT_Transcripts"        # on‑Drive if mounted
                if HOME_DRIVE.exists()
                else Path("YT_Transcripts"))         # local fallback
SAVE_DIR.mkdir(parents=True, exist_ok=True)

_ID_REGEX = re.compile(r"(?:v=|\/)([0-9A-Za-z_-]{11})")

# ──────────────────────────────────────────────────────────────────────────────
def extract_id(token: str) -> str:
    """Return the 11‑char YouTube video ID from any common URL or raw ID."""
    m = _ID_REGEX.search(token)
    if m:
        return m.group(1)
    if len(token) == 11:
        return token
    raise ValueError(f"🔴 Can't parse a YouTube ID from ‘{token}’")

def download_audio(video_id: str) -> Path:
    """Grab best‑quality m4a audio with yt‑dlp and return its Path."""
    url     = f"https://www.youtube.com/watch?v={video_id}"
    out_f   = Path(f"{video_id}.m4a")
    ydl_ops = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(out_f),
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    }
    print(f"📥 {video_id} – downloading …")
    yt_dlp.YoutubeDL(ydl_ops).download([url])
    if not out_f.exists():
        raise RuntimeError(f"yt‑dlp failed for {video_id}")
    return out_f

print(f"🚀 Loading faster‑whisper‑{MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

def transcribe(audio_path: Path, vid: str):
    print(f"🧠 {vid} – transcribing …")
    segments, _ = model.transcribe(str(audio_path), beam_size=1)  # greedy = fastest
    text = "".join(seg.text for seg in segments)

    out_file = SAVE_DIR / f"{vid}_transcript.txt"
    out_file.write_text(text, encoding="utf‑8")
    print(f"✅ saved → {out_file}")

    audio_path.unlink(missing_ok=True)          # tidy up tmp audio

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python multiple_youtube_transcripter.py <url_or_id> …")

    for raw in sys.argv[1:]:
        try:
            vid = extract_id(raw)
            transcribe(download_audio(vid), vid)
        except Exception as exc:
            print(f"❌ {raw}: {exc}")

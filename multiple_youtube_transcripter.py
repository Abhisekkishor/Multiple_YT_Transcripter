import os
import sys
import whisper
import yt_dlp

# Choose the Whisper model - tiny is faster, base is slightly more accurate
MODEL_SIZE = "tiny"  # Change to "small" for slightly better results
SAVE_DIR = "Multiple_YT_Transcripts"

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def download_audio(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    filename = f"{video_id}.m4a"
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': filename,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
    }

    print(f"📥 Downloading audio for video ID: {video_id}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Audio file '{filename}' not found.")
    return filename

def transcribe_audio(audio_file, video_id):
    print(f"🧠 Transcribing {audio_file} with Whisper ({MODEL_SIZE})...")
    model = whisper.load_model(MODEL_SIZE)
    result = model.transcribe(audio_file)
    transcript_text = result['text']

    output_file = os.path.join(SAVE_DIR, f"{video_id}_transcript.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    print(f"✅ Transcript saved to: {output_file}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multiple_youtube_transcripter.py <video_id_1> <video_id_2> ...")
        sys.exit(1)

    video_ids = sys.argv[1:]

    for vid in video_ids:
        try:
            audio_path = download_audio(vid)
            transcribe_audio(audio_path, vid)
        except Exception as e:
            print(f"❌ Failed to process {vid}: {str(e)}\n")

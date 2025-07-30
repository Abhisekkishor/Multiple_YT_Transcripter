# Multiple YT Transcripter – GPU edition

## Quick start in Colab

1. **Open a new GPU notebook** – Runtime ▸ Change runtime type ▸ GPU.
2. Paste the One‑Cell launcher (below) into the first cell and run.
3. When prompted, paste one or more YouTube URLs or IDs (comma/space separated).
4. Transcripts land in  
   • `/content/drive/MyDrive/YT_Transcripts/…` if Drive is mounted,  
   • otherwise in the notebook’s working directory.

## Local CLI

```bash
pip install -r requirements.txt
python multiple_youtube_transcripter.py https://youtu.be/dQw4w9WgXcQ 8zEHfIGQI9M

import os
import json
import whisper
import subprocess

VIDEO_DIR = r"D:\SumMe"
ASR_DIR = "data/asr"

os.makedirs(ASR_DIR, exist_ok=True)

# Load Whisper model
model = whisper.load_model("base")

# Convert non-MP4 videos to MP4
def convert_to_mp4(video_path):
    mp4_path = os.path.splitext(video_path)[0] + ".mp4"
    
    if not os.path.exists(mp4_path):  # only convert if not already done
        print(f"Converting: {video_path} → {mp4_path}")
        subprocess.run([
            "ffmpeg", "-i", video_path, "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", mp4_path
        ])
    
    return mp4_path

# Function to run ASR (speech-to-text)
def run_asr(video_path):
    result = model.transcribe(
        video_path,
        verbose=False,
        fp16=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0
    )

    # Clean and return the ASR segments
    clean_segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        }
        for seg in result["segments"]
    ]

    return clean_segments

# Process each video file
for video_file in os.listdir(VIDEO_DIR):
    # Skip non-video files
    if not video_file.endswith((".mp4", ".mkv", ".webm")):
        continue

    video_path = os.path.join(VIDEO_DIR, video_file)
    video_name = os.path.splitext(video_file)[0]
    output_path = os.path.join(ASR_DIR, f"{video_name}.json")

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"Skipping {video_name}")
        continue

    print(f"Processing ASR: {video_name}")

    try:
        # If it's not MP4, convert it to MP4
        if not video_path.endswith(".mp4"):
            video_path = convert_to_mp4(video_path)

        # Run ASR on the (converted) video file
        segments = run_asr(video_path)

        # Save the ASR output as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

    except Exception as e:
        print(f"Error in {video_name}: {e}")
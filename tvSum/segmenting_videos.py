import os
import re
import json
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# -----------------------------
# Config
# -----------------------------
VIDEO_DIR = r"D:\SumMe"
OUTPUT_DIR = "data/SumMe_shots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Clean filename (VERY IMPORTANT)
# -----------------------------
def clean_name(name):
    # Remove special characters and extra spaces
    return re.sub(r"[^\w\-. ]", "", name).strip()


# -----------------------------
# Convert non-MP4 videos to MP4
# -----------------------------
def convert_to_mp4(video_path):
    mp4_path = os.path.splitext(video_path)[0] + ".mp4"
    
    if not os.path.exists(mp4_path):  # only convert if not already done
        print(f"Converting: {video_path} → {mp4_path}")
        subprocess.run([
            "ffmpeg", "-i", video_path, "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", mp4_path
        ])
    
    return mp4_path


# -----------------------------
# Process videos
# -----------------------------
for video_file in os.listdir(VIDEO_DIR):

    # Skip non-video files
    if not video_file.endswith((".mp4", ".mkv", ".webm")):
        print(f"Skipping non-video file: {video_file}")
        continue

    video_path = os.path.join(VIDEO_DIR, video_file)
    raw_name = os.path.splitext(video_file)[0]
    video_name = clean_name(raw_name)

    print(f"\nProcessing: {video_name}")

    # create output folder per video
    video_out_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    output_json = os.path.join(video_out_dir, "shots.json")

    # Skip already processed videos
    if os.path.exists(output_json):
        print("Already processed, skipping...")
        continue

    try:
        # If it's not MP4, convert it to MP4
        if not video_path.endswith(".mp4"):
            video_path = convert_to_mp4(video_path)

        # -----------------------------
        # Scene detection
        # -----------------------------
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scenes = scene_manager.get_scene_list()

        # -----------------------------
        # Save scenes
        # -----------------------------
        scene_data = []

        for i, scene in enumerate(scenes):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()

            scene_data.append({
                "shot_id": i,
                "start": start_time,
                "end": end_time
            })

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2)

        print(f"Saved {len(scene_data)} shots")

        # IMPORTANT: release memory
        video_manager.release()

    except Exception as e:
        print(f"Error processing {video_name}: {e}")
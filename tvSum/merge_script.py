import os
import json
import re

# --------------------------------------------------
# Paths
# --------------------------------------------------
SEGMENT_DIR = "data/SumMe_shots"
ASR_DIR = "data/asr"
OUTPUT_DIR = "data/merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Clean filename (same logic as segmentation)
# --------------------------------------------------
def clean_name(name):
    return re.sub(r"[^\w\-. ]", "", name).strip()


# --------------------------------------------------
# Get ASR text overlapping with shot
# --------------------------------------------------
def get_asr_for_shot(start, end, segments):
    texts = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # overlap condition
        if seg_start < end and seg_end > start:
            texts.append(seg["text"])

    return " ".join(texts).strip()


# --------------------------------------------------
# Build ASR lookup (clean_name -> actual filename)
# --------------------------------------------------
asr_lookup = {}

for f in os.listdir(ASR_DIR):
    if f.endswith(".json"):
        base = os.path.splitext(f)[0]
        clean_base = clean_name(base)
        asr_lookup[clean_base] = f


# --------------------------------------------------
# Merge loop
# --------------------------------------------------
for video_name in os.listdir(SEGMENT_DIR):

    video_path = os.path.join(SEGMENT_DIR, video_name)

    # skip non-directories
    if not os.path.isdir(video_path):
        continue

    clean_video_name = clean_name(video_name)

    shots_path = os.path.join(video_path, "shots.json")

    if clean_video_name not in asr_lookup:
        print(f"⚠️ Skipping {video_name} (no matching ASR)")
        continue

    asr_filename = asr_lookup[clean_video_name]
    asr_path = os.path.join(ASR_DIR, asr_filename)

    if not os.path.exists(shots_path):
        print(f"⚠️ Skipping {video_name} (missing shots.json)")
        continue

    print(f"✅ Merging: {video_name}")

    # -----------------------------
    # Load data
    # -----------------------------
    try:
        with open(shots_path, "r") as f:
            shots = json.load(f)

        with open(asr_path, "r") as f:
            segments = json.load(f)

    except Exception as e:
        print(f"❌ Error reading {video_name}: {e}")
        continue

    # -----------------------------
    # Merge shots + ASR
    # -----------------------------
    merged = []

    for shot in shots:
        start = shot["start"]
        end = shot["end"]

        asr_text = get_asr_for_shot(start, end, segments)

        if not asr_text:
            asr_text = "no speech"

        merged.append({
            "shot_id": shot["shot_id"],
            "start": start,
            "end": end,
            "asr": asr_text
        })

    # -----------------------------
    # Save output
    # -----------------------------
    out_path = os.path.join(OUTPUT_DIR, f"{clean_video_name}.json")

    try:
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)

        print(f"💾 Saved: {out_path}")

    except Exception as e:
        print(f"❌ Error saving {video_name}: {e}")
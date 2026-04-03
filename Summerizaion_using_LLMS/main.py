import os
from file_exists import load_json_data, process_video_shots
from summary import select_shots_knapsack
from combine_shots import combine_shots
from Evaluation.coverage import compute_coverage
from Evaluation.redundancy import compute_redundancy
import json
# -------------------------
# HELPERS
# -------------------------
def get_raw_videos(raw_videos_dir="data/raw_videos"):
    return [
        os.path.join(raw_videos_dir, f)
        for f in os.listdir(raw_videos_dir)
        if f.lower().endswith(".mp4")
    ]


def video_to_json_path(video_path, segmentation_dir="data/segmentation"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        segmentation_dir,
        video_name,
        f"{video_name}.json"
    )

# -------------------------
# PIPELINE
# -------------------------
raw_videos = get_raw_videos()

results_dir = "data/llm_results"
os.makedirs(results_dir, exist_ok=True)
for video_path in raw_videos:
    print(f"\n🎬 Processing: {video_path}")

    json_path = video_to_json_path(video_path)

    if not os.path.exists(json_path):
        print(f"⚠️ JSON not found, skipping: {json_path}")
        continue

    # 1. Load shot evidence
    video_data = load_json_data(json_path)
    num_shots = len(video_data)

    # 2. Score shots with LLaMA
    scored_shots = process_video_shots(video_data)

    # 3. Select best shots (knapsack)
    selected_shots = select_shots_knapsack(
        scored_shots,
        max_duration=6.0
    )
    print(f"✅ Selected {selected_shots} shots for summary (")

    # 4. Coverage
    original_shots = list(video_data.values())
    coverage = compute_coverage(original_shots, selected_shots)

    # 5. Redundancy
    redundancy = compute_redundancy(selected_shots)

    print(f"📊 Coverage:   {coverage:.4f}")
    print(f"🔁 Redundancy: {redundancy:.4f}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    results_path = os.path.join(results_dir, f"{video_name}_results.json")

    results_data = {
        "num_shots": num_shots,
        "scored_shots": scored_shots,
        "selected_shots": selected_shots,
        "coverage": coverage,
        "redundancy": redundancy
    }
    print(f"💾 Saving results to {results_data}...")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)

    print(f"💾 Saved results to {results_path}")


    # 6. Build summary video
    # combine_shots(video_path, selected_shots)



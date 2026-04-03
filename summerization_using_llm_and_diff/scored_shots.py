import os
import json
import torch
import numpy as np

from tvSum.diffusion_model import MaskDiffusionModel


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Paths
# --------------------------------------------------
SEGMENTATION_DIR = "../data/segmentation"
IMAGE_EMB_DIR = "../data/embeddings"
TEXT_EMB_DIR = "../data/text_embeddings"
MODEL_PATH = "diffusion_model.pth"


# --------------------------------------------------
# Knapsack
# --------------------------------------------------
def select_shots_knapsack(scored_shots, max_duration=5.0):

    shots = sorted(
        scored_shots,
        key=lambda x: x["score"] / max(x["data"]["end"] - x["data"]["start"], 0.01),
        reverse=True
    )

    selected = []
    total_time = 0.0

    for shot in shots:

        duration = max(shot["data"]["end"] - shot["data"]["start"], 0.01)

        if total_time + duration <= max_duration:
            selected.append(shot)
            total_time += duration

    selected.sort(key=lambda x: x["data"]["start"])

    return selected


# --------------------------------------------------
# Load model
# --------------------------------------------------
print("Loading diffusion model...")

model = MaskDiffusionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded.")


# --------------------------------------------------
# Process videos
# --------------------------------------------------
for video_folder in os.listdir(SEGMENTATION_DIR):

    video_path = os.path.join(SEGMENTATION_DIR, video_folder)

    if not os.path.isdir(video_path):
        continue

    shots_file = os.path.join(video_path, "shots.json")

    if not os.path.exists(shots_file):
        print("Skipping", video_folder, "(no shots.json)")
        continue

    print("\nProcessing video:", video_folder)


    # --------------------------------------------------
    # Load shots
    # --------------------------------------------------
    with open(shots_file) as f:
        shots_dict = json.load(f)

    shots = [shots_dict[k] for k in sorted(shots_dict.keys(), key=lambda x: int(x))]

    print("Number of shots:", len(shots))


    # --------------------------------------------------
    # Load IMAGE embeddings
    # --------------------------------------------------
    image_file = os.path.join(IMAGE_EMB_DIR, f"{video_folder}.npz")

    if not os.path.exists(image_file):
        print("Image embeddings missing, skipping.")
        continue

    image_npz = np.load(image_file)

    image_list = []

    for key in sorted(image_npz.files, key=lambda x: int(x)):
        image_list.append(image_npz[key])

    image_embeddings = np.stack(image_list)

    print("Image embeddings:", image_embeddings.shape)


    # --------------------------------------------------
    # Load TEXT embeddings
    # --------------------------------------------------
    text_file = os.path.join(TEXT_EMB_DIR, f"{video_folder}.npy")

    if not os.path.exists(text_file):
        print("Text embeddings missing, skipping.")
        continue

    text_embeddings = np.load(text_file)

    print("Text embeddings:", text_embeddings.shape)


    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------
    if len(image_embeddings) != len(text_embeddings):

        min_len = min(len(image_embeddings), len(text_embeddings))

        print("Mismatch detected. Truncating to", min_len)

        image_embeddings = image_embeddings[:min_len]
        text_embeddings = text_embeddings[:min_len]
        shots = shots[:min_len]


    # --------------------------------------------------
    # Combine embeddings
    # --------------------------------------------------
    combined = np.concatenate([image_embeddings, text_embeddings], axis=1)

    E = torch.tensor(combined).float()

    E = E.unsqueeze(0).to(device)

    B, N, D = E.shape

    print("Combined embedding shape:", E.shape)


    # --------------------------------------------------
    # Diffusion inputs
    # --------------------------------------------------
    x_t = torch.zeros(B, N).to(device)
    t = torch.zeros(B, dtype=torch.long).to(device)


    # --------------------------------------------------
    # Run model
    # --------------------------------------------------
    with torch.no_grad():
        scores = model(x_t, E, t)

    scores = torch.sigmoid(scores).cpu().view(-1).tolist()

    print("Scores predicted:", len(scores))


    # --------------------------------------------------
    # Build scored shots
    # --------------------------------------------------
    scored_shots = []

    for shot, score in zip(shots, scores):

        scored_shots.append({
            "score": float(score),
            "data": shot
        })


    # --------------------------------------------------
    # Knapsack
    # --------------------------------------------------
    selected = select_shots_knapsack(scored_shots, max_duration=5.0)

    print("Selected shots:", len(selected))


    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------
    output_file = os.path.join(video_path, "diffusion_summary.json")

    with open(output_file, "w") as f:
        json.dump(selected, f, indent=4)

    print("Saved summary →", output_file)


print("\nAll videos processed.")
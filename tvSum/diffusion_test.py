import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from diffusion_model import MaskDiffusionModel


# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "../data/segmentation"
OUTPUT_DIR = "summaries"
MODEL_PATH = "best_diffusion_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTEPS = 100
MAX_SUMMARY_DURATION = 5.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# LOAD MODELS
# -------------------------
print("Loading models...")

text_model = SentenceTransformer("all-mpnet-base-v2")

diffusion_model = MaskDiffusionModel(
    embedding_dim=768,
    hidden_dim=512,
    timesteps=TIMESTEPS
).to(DEVICE)

diffusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
diffusion_model.eval()

print("Models loaded.")


# -------------------------
# DIFFUSION SAMPLING
# -------------------------
@torch.no_grad()
def sample_scores(model, E, dur, timesteps=100):

    device = next(model.parameters()).device

    # ensure correct shapes
    if len(E.shape) == 1:
        E = np.expand_dims(E, axis=0)

    if len(dur.shape) == 0:
        dur = np.expand_dims(dur, axis=0)

    E = torch.tensor(E).unsqueeze(0).to(device).float()   # [1, N, D]
    dur = torch.tensor(dur).unsqueeze(0).to(device).float()  # [1, N]

    N = E.shape[1]

    x = torch.randn(1, N, device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        noise_pred = model(x, E, dur, t_tensor)

        x = x - noise_pred / timesteps

    return x.squeeze(0).cpu().numpy()


# -------------------------
# KNAPSACK
# -------------------------
def select_shots_knapsack(shots, max_duration=5.0):

    shots = sorted(
        shots,
        key=lambda x: x["score"] / max(x["end"] - x["start"], 0.01),
        reverse=True
    )

    selected = []
    total_time = 0.0

    for shot in shots:
        duration = shot["end"] - shot["start"]

        if total_time + duration <= max_duration:
            selected.append(shot)
            total_time += duration

    selected.sort(key=lambda x: x["start"])
    return selected


# -------------------------
# MAIN LOOP
# -------------------------
print("Dataset dir:", os.path.abspath(DATASET_DIR))

for video_folder in os.listdir(DATASET_DIR):

    video_path = os.path.join(DATASET_DIR, video_folder)

    if not os.path.isdir(video_path):
        continue

    json_path = os.path.join(video_path, f"{video_folder}.json")

    if not os.path.exists(json_path):
        print("Missing JSON:", json_path)
        continue

    print(f"\nProcessing video: {video_folder}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots = sorted(data.items(), key=lambda x: int(x[0]))

    texts = []
    durations = []
    shot_list = []

    # -------------------------
    # BUILD VALID SHOTS
    # -------------------------
    for _, shot in shots:

        start = shot.get("start", 0)
        end = shot.get("end", 0)

        duration = end - start

        # skip invalid shots
        if duration <= 0:
            continue

        asr = shot.get("asr_text", "")
        caption = shot.get("visual_caption", "")

        text = f"{asr}. {caption}".strip()

        texts.append(text)
        durations.append(duration)

        shot_list.append({
            "shot_id": shot.get("shot_id", 0),
            "start": start,
            "end": end,
            "asr": asr
        })

    # -------------------------
    # SKIP EMPTY VIDEOS
    # -------------------------
    if len(texts) == 0:
        print("⚠️ Skipping empty video:", video_folder)
        continue

    print(f"Valid shots: {len(texts)}")

    # -------------------------
    # EMBEDDINGS
    # -------------------------
    embeddings = text_model.encode(texts)

    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    durations = np.array(durations)

    # -------------------------
    # DIFFUSION
    # -------------------------
    scores = sample_scores(diffusion_model, embeddings, durations)

    # normalize
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # attach scores
    for i in range(len(shot_list)):
        shot_list[i]["score"] = float(scores[i])

    # -------------------------
    # SELECT SUMMARY
    # -------------------------
    summary = select_shots_knapsack(
        shot_list,
        max_duration=MAX_SUMMARY_DURATION
    )

    # -------------------------
    # SAVE
    # -------------------------
    output_path = os.path.join(OUTPUT_DIR, f"{video_folder}.json")

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {output_path}")


print("\nAll summaries generated.")
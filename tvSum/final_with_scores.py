import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"
INPUT_DIR = "data/final_with_scores"
OUTPUT_DIR = "data/text_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)


def load_texts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        shots = json.load(f)

    texts = []

    for shot in shots:
        asr = shot.get("asr", "").strip()

        if not asr:
            asr = "no speech"

        texts.append(asr)

    return texts


for file in os.listdir(INPUT_DIR):
    if not file.endswith(".json"):
        continue

    input_path = os.path.join(INPUT_DIR, file)
    video_name = os.path.splitext(file)[0]

    print(f"Processing: {video_name}")

    texts = load_texts(input_path)
    embeddings = model.encode(texts, convert_to_numpy=True)

    save_path = os.path.join(OUTPUT_DIR, f"{video_name}.npy")
    np.save(save_path, embeddings)

    print(f"Saved: {save_path}")

print("All text embeddings generated.")
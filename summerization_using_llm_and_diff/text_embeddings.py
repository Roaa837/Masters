import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-mpnet-base-v2")


SEGMENTATION_DIR = "../data/segmentation"
OUTPUT_DIR = "../data/text_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_text(json_path):

    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # shots are stored with keys "1","2","3"...
    shots = sorted(data.items(), key=lambda x: int(x[0]))

    texts = []

    for shot_id, shot_data in shots:

        asr = shot_data.get("asr_text", "")
        caption = shot_data.get("visual_caption", "")

        combined = f"ASR: {asr}. VISUAL: {caption}."

        texts.append(combined)

    return texts


# --------------------------------------------------
# Iterate through segmentation folders
# --------------------------------------------------
for video_folder in os.listdir(SEGMENTATION_DIR):

    video_path = os.path.join(SEGMENTATION_DIR, video_folder)

    if not os.path.isdir(video_path):
        continue

    caption_json = os.path.join(video_path, f"{video_folder}.json")

    if not os.path.exists(caption_json):
        print("No captions for:", video_folder)
        continue

    print("Processing captions:", video_folder)

    texts = encode_text(caption_json)

    embeddings = model.encode(texts)

    save_path = os.path.join(OUTPUT_DIR, f"{video_folder}.npy")

    np.save(save_path, embeddings)

    print("Saved text embeddings:", save_path)


print("\nAll text embeddings generated.")
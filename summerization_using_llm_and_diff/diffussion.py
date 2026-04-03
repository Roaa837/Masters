from image_embeddings import extract_image_embeddings
from text_embeddings import vectorize_texts
import torch
import json
def prepering_inputs_for_diffussion(json_path,keyframes_path):
    # 1. extract text embeddings
    text_embeddings = vectorize_texts(json_path)
    # 2. extract image embeddings
    image_embeddings = extract_image_embeddings(keyframes_path)
    fused = torch.cat([text_embeddings, image_embeddings], dim=1)
    return fused
    
def extract_durations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    shots = sorted(data.values(), key=lambda x: x["shot_id"])

    durations = [shot["end"] - shot["start"] for shot in shots]
    return torch.tensor(durations)

def building_mask_from_results(json_results_path):
    with open(json_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    N = results["num_shots"]
    selected_shots = results["selected_shots"]
    mask = torch.zeros(N, dtype=torch.float32)
    for shot in selected_shots:
        shot_id = int(shot["shot_id"])
        mask[shot_id-1] = 1.0
    return mask


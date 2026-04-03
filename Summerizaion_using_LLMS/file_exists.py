import os
import json
import requests
from typing import Dict, Any, List

# -------------------------
# CONFIG
# -------------------------
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

VIDEO_DIR = os.path.join(
    "data",
    "segmentation",
    "_Floral_ by ADIDAS - AI Spec AD"
)

JSON_PATH = os.path.join(
    VIDEO_DIR,
    "_Floral_ by ADIDAS - AI Spec AD.json"
)

# -------------------------
# UTILS
# -------------------------
def load_json_data(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"].strip()


# -------------------------
# SCORING
# -------------------------
def score_shot_with_llama(shot: Dict[str, Any]) -> float:
    """
    Scores a shot from 0–10 using LLaMA-3.1.
    Returns a float. Falls back safely if parsing fails.
    """

    prompt = f"""
You are evaluating a shot from a marketing video.

Return ONLY a single number between 0 and 10.
No explanation. No text.

Scoring criteria:
- Emotional impact
- Brand visibility
- Message clarity
- Visual uniqueness
- Importance for a short summary

ASR:
{shot['asr_text']}

Visual description:
{shot['visual_caption']}

Score:
"""

    response = ollama_generate(prompt)

    try:
        return float(response)
    except ValueError:
        print("⚠️ Invalid score returned:", response)
        return 0.0


# -------------------------
# PIPELINE
# -------------------------
def process_video_shots(video_data: Dict[str, Any]) -> List:
    scored_shots = []

    for shot_id, shot_data in video_data.items():
        score = score_shot_with_llama(shot_data)
        scored_shots.append({
            "shot_id": shot_id,
            "score": score,
            "data": shot_data
        })

    scored_shots.sort(key=lambda x: x["score"], reverse=True)
    return scored_shots


# -------------------------
# MAIN
# -------------------------
# if __name__ == "__main__":
#     video_data = load_json_data(JSON_PATH)

#     scored_shots = process_video_shots(video_data)

# print(scored_shots)

import os
import numpy as np

from utils.io import load_embeddings, load_summary
from metrics.coverage import compute_coverage
from metrics.diversity import compute_diversity
from metrics.representativeness import compute_representativeness


# ----------------------------------------
# Extract embeddings from summary JSON
# ----------------------------------------
def extract_embeddings(summary, video_embeds):

    indices = []

    # LLM format
    if isinstance(summary, dict):
        shots = summary.get("selected_shots", [])

        for s in shots:
            shot_id = int(s["data"]["shot_id"]) - 1
            indices.append(shot_id)

    # Diffusion format
    elif isinstance(summary, list):
        for s in summary:
            shot_id = int(s["shot_id"]) - 1
            indices.append(shot_id)

    else:
        raise ValueError("Unknown summary format")

    # safe indexing
    indices = [i for i in indices if 0 <= i < len(video_embeds)]

    if len(indices) == 0:
        return np.empty((0, video_embeds.shape[1]))

    return video_embeds[indices]


# ----------------------------------------
# Evaluate ONE video
# ----------------------------------------
def evaluate(video_name):

    llm_path = os.path.join("data/llm_results", f"{video_name}_results.json")

    # ✅ FIXED PATH (VERY IMPORTANT)
    diffusion_path = os.path.join("tvSum", "summaries", f"{video_name}.json")

    embeddings_path = os.path.join("data/embeddings", f"{video_name}.npz")

    if not os.path.exists(llm_path):
        print(f"❌ Missing LLM file for {video_name}")
        return None

    if not os.path.exists(diffusion_path):
        print(f"❌ Missing diffusion file for {video_name}")
        return None

    if not os.path.exists(embeddings_path):
        print(f"❌ Missing embeddings for {video_name}")
        return None

    llm_summary = load_summary(llm_path)
    diff_summary = load_summary(diffusion_path)
    video_embeds = load_embeddings(embeddings_path)

    if len(video_embeds) == 0:
        print(f"❌ Empty embeddings for {video_name}")
        return None

    llm_embeds = extract_embeddings(llm_summary, video_embeds)
    diff_embeds = extract_embeddings(diff_summary, video_embeds)

    if len(llm_embeds) == 0:
        print(f"⚠️ Empty LLM summary for {video_name}")
        return None

    if len(diff_embeds) == 0:
        print(f"⚠️ Empty diffusion summary for {video_name}")
        return None

    results = {}

    results["llm"] = {
        "coverage": compute_coverage(video_embeds, llm_embeds),
        "diversity": compute_diversity(llm_embeds),
        "representativeness": compute_representativeness(video_embeds, llm_embeds)
    }

    results["diffusion"] = {
        "coverage": compute_coverage(video_embeds, diff_embeds),
        "diversity": compute_diversity(diff_embeds),
        "representativeness": compute_representativeness(video_embeds, diff_embeds)
    }

    return results


# ----------------------------------------
# Run evaluation
# ----------------------------------------
if __name__ == "__main__":

    BASE_PATH = "data/segmentation"

    videos = [
        f for f in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, f))
    ]

    print("📁 Videos found:", videos)

    all_results = []

    for v in videos:
        print(f"\n🔍 Processing: {v}")

        res = evaluate(v)

        if res is None:
            continue

        print("Results:")
        print(res)

        all_results.append(res)

    # ----------------------------------------
    # Averages
    # ----------------------------------------
    if len(all_results) == 0:
        print("\n❌ No valid results.")
    else:

        llm_cov, llm_div, llm_rep = [], [], []
        diff_cov, diff_div, diff_rep = [], [], []

        for r in all_results:
            llm_cov.append(float(r["llm"]["coverage"]))
            llm_div.append(float(r["llm"]["diversity"]))
            llm_rep.append(float(r["llm"]["representativeness"]))

            diff_cov.append(float(r["diffusion"]["coverage"]))
            diff_div.append(float(r["diffusion"]["diversity"]))
            diff_rep.append(float(r["diffusion"]["representativeness"]))

        print("\n📊 FINAL AVERAGES -------------------")

        print("\nLLM:")
        print("Coverage:", np.mean(llm_cov))
        print("Diversity:", np.mean(llm_div))
        print("Representativeness:", np.mean(llm_rep))

        print("\nDiffusion:")
        print("Coverage:", np.mean(diff_cov))
        print("Diversity:", np.mean(diff_div))
        print("Representativeness:", np.mean(diff_rep))
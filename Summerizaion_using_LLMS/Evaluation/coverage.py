# “For every original shot, is there at least one summary shot that is similar to it"
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_coverage(original_shots, summary_shots):
    original_texts = [
        s["asr_text"] + " " + s["visual_caption"]
        for s in original_shots
    ]
    summary_texts = [
        s["data"]["asr_text"] + " " + s["data"]["visual_caption"]
        for s in summary_shots
    ]

    E = model.encode(original_texts, normalize_embeddings=True)
    S = model.encode(summary_texts, normalize_embeddings=True)

    coverage_scores = []
    for e in E:
        sims = np.dot(S, e)
        coverage_scores.append(sims.max())

    return float(np.mean(coverage_scores))


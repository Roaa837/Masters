import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def compute_redundancy(summary_shots):
    texts = [
        s["data"]["asr_text"] + " " + s["data"]["visual_caption"]
        for s in summary_shots
    ]

    S = model.encode(texts, normalize_embeddings=True)

    sims = np.dot(S, S.T)

    n = len(summary_shots)
    if n < 2:
        return 0.0

    # exclude diagonal
    redundancy = (sims.sum() - n) / (n * (n - 1))
    return float(redundancy)

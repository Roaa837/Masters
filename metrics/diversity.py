import numpy as np
from utils.similarity import cosine_similarity

def compute_diversity(summary_embeds):
    if len(summary_embeds) < 2:
        return 0.0

    sim = cosine_similarity(summary_embeds, summary_embeds)
    n = len(summary_embeds)

    # remove diagonal
    diversity = 1 - (np.sum(sim) - n) / (n * (n - 1))
    return diversity
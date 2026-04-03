import numpy as np
from utils.similarity import cosine_similarity

def compute_coverage(video_embeds, summary_embeds):
    """
    video_embeds: (N, D)
    summary_embeds: (M, D)
    """
    sim = cosine_similarity(video_embeds, summary_embeds)
    max_sim = np.max(sim, axis=1)
    return np.mean(max_sim)
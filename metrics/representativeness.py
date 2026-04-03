from sklearn.cluster import KMeans
import numpy as np

def compute_representativeness(video_embeds, summary_embeds, k=10):
    n_samples = len(video_embeds)

    # make sure k is valid
    k = min(k, n_samples)

    # avoid k=0 or k=1 issues
    if k < 2:
        return 0.0

    kmeans = KMeans(n_clusters=k, random_state=0).fit(video_embeds)
    clusters = kmeans.predict(summary_embeds)

    coverage = len(set(clusters)) / k
    return coverage
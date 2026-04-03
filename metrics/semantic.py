from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(full_text, summary_text):
    emb1 = model.encode([full_text])[0]
    emb2 = model.encode([summary_text])[0]

    sim = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
    )

    return sim
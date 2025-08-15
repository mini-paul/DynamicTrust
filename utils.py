# utils.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# It's efficient to load the model once and reuse it.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculates the cosine similarity between two texts using sentence embeddings.
    """
    if not text1 or not text2:
        return 0.0

    # Generate embeddings for both texts
    embeddings = embedding_model.encode([text1, text2])

    # --- FIX START ---
    # 1. Access the individual embeddings using standard list/array indexing.
    #    The incorrect dot notation (embeddings.[1]) is removed.
    embedding1 = embeddings[0].reshape(1, -1)
    embedding2 = embeddings[1].reshape(1, -1)

    # 2. The result of cosine_similarity is a 2D array (e.g., [[0.95]]),
    #    so we need to extract the single float value from it using [0, 0].
    sim_score = cosine_similarity(embedding1, embedding2)[0, 0]
    # --- FIX END ---

    return float(sim_score)
# Simple Python client-style wrapper representing Endee Vector DB

import numpy as np

class EndeeVectorDB:
    def __init__(self):
        print("Connected to Endee Vector Database (simulated client).")
        self.text_store = []
        self.embedding_store = []

    def store_embeddings(self, texts, embeddings):
        print(f"Storing {len(texts)} documents in Endee...")
        self.text_store.extend(texts)
        self.embedding_store.extend(embeddings)
        print("Stored successfully in Endee.")

    def search(self, query_embedding, top_k=2):
        print("Performing vector similarity search in Endee...")

        # Compute cosine similarity (simple version)
        similarities = []
        for emb in self.embedding_store:
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(sim)

        # Get top-k most similar results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [self.text_store[i] for i in top_indices]
        return results

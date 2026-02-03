from sentence_transformers import SentenceTransformer
from vector_db import EndeeVectorDB
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load stored data (simulating Endee retrieval)
embeddings = np.load("my_ai_project/embeddings.npy", allow_pickle=True)
texts = np.load("my_ai_project/texts.npy", allow_pickle=True)

# Connect to Endee (simulated)
db = EndeeVectorDB()
db.text_store = list(texts)
db.embedding_store = list(embeddings)

# ---- USER INPUT SEARCH (NEW FEATURE) ----
query = input("\nEnter your question: ")

# Convert query to embedding
query_embedding = model.encode(query)

# Search in Endee
results = db.search(query_embedding, top_k=2)

print("\n🔍 Most Relevant Results:")
for r in results:
    print("-", r)

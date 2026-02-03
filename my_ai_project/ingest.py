from sentence_transformers import SentenceTransformer
from vector_db import EndeeVectorDB
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents (your "database")
documents = [
    "Python is a programming language.",
    "Machine learning is a part of AI.",
    "Vector databases store embeddings.",
    "Endee is a high-performance vector database."
]

# Convert documents to embeddings
embeddings = model.encode(documents)

print("\n✅ Documents converted to embeddings successfully!")

# Connect to Endee and store data
db = EndeeVectorDB()
db.store_embeddings(documents, embeddings)

# Save embeddings locally (so search.py can use them)
np.save("my_ai_project/embeddings.npy", embeddings)
np.save("my_ai_project/texts.npy", documents)

print("✅ Ingestion completed and data stored.")

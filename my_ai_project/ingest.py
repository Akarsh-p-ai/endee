# Simple script to store text in Endee (Beginner Friendly)

from sentence_transformers import SentenceTransformer

# This is just a placeholder to show structure (we will connect Endee later)
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Python is a programming language.",
    "Machine learning is a part of AI.",
    "Vector databases store embeddings."
]

embeddings = model.encode(documents)

print("Documents converted to embeddings successfully!")
print(embeddings)

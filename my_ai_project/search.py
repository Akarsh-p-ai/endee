from sentence_transformers import SentenceTransformer
from vector_db import EndeeVectorDB
import numpy as np
import re

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load stored data (simulating Endee retrieval)
embeddings = np.load("embeddings.npy", allow_pickle=True)
texts = np.load("texts.npy", allow_pickle=True)

# Connect to Endee (simulated)
db = EndeeVectorDB()
db.text_store = list(texts)
db.embedding_store = list(embeddings)

# ---- USER INPUT SEARCH (NEW FEATURE) ----
query = input("\nEnter your question: ")

# Convert query to embedding
query_embedding = model.encode(query)

# Search in Endee
results = db.search(query_embedding, top_k=15)

# Calculate similarity scores with intelligent keyword boosting
from sklearn.metrics.pairwise import cosine_similarity

# Extract keywords from query (remove only common words)
stop_words = {'a', 'an', 'the', 'is', 'are', 'of', 'in', 'to', 'find', 'me', 'which', 'near', 'at', 'for', 'and', 'or', 'with', 'by', 'from', 'was', 'where', 'do', 'does', 'can', 'be'}
query_words = [word for word in query.lower().split() if word not in stop_words and len(word) > 2]

# Track if any important query words were found
important_query_words = query_words.copy()

# Keywords with boost strength (0-1 scale)
boosting_rules = {
    'water': {'keywords': ['sea', 'ocean', 'coastal', 'gulf', 'beach', 'shore'], 'boost': 0.25},
    'river': {'keywords': ['river', 'riverside', 'flowing', 'banks', 'ganges', 'godavari', 'bhima'], 'boost': 0.10},
    'mountain': {'keywords': ['mountain', 'hill', 'himalaya', 'sahyadri', 'peak', 'elevation', 'altitude'], 'boost': 0.12},
    'island': {'keywords': ['island', 'peninsula', 'isle'], 'boost': 0.15},
    'cave': {'keywords': ['cave', 'cavern', 'ellora', 'carved'], 'boost': 0.10},
    'statue': {'keywords': ['statue', 'monolithic', 'tall', 'monument', 'ft', 'feet'], 'boost': 0.12},
    'ancient': {'keywords': ['ancient', 'oldest', 'first', 'established', 'century'], 'boost': 0.10},
    'festival': {'keywords': ['festival', 'fair', 'mela', 'celebrated'], 'boost': 0.12},
}

# Negative keywords (penalize if query is about specific water and result has different geography)
negative_keywords = {
    'sea': {'avoid': ['cave', 'hill', 'mountain', 'himalaya'], 'penalty': 0.15},
    'mountain': {'avoid': ['sea', 'ocean', 'beach', 'coastal'], 'penalty': 0.10},
    'cave': {'avoid': ['sea', 'ocean', 'beach'], 'penalty': 0.10},
}

result_scores = []
for i, result in enumerate(results, 1):
    # Calculate embedding similarity score
    result_embedding = model.encode(result)
    similarity = cosine_similarity([query_embedding], [result_embedding])[0][0]
    
    # Start with base similarity
    boosted_score = similarity
    result_lower = result.lower()
    
    # CRITICAL: Check if ANY important query words are found in result
    found_query_words = 0
    for query_word in important_query_words:
        if query_word in result_lower:
            found_query_words += 1
            boosted_score += 0.20  # Boost heavily if query word found
    
    # PENALTY: If query has specific keywords but result doesn't mention them
    if len(important_query_words) > 0 and found_query_words == 0:
        boosted_score *= 0.5  # Cut score in half if NO query words found
    
    # Apply semantic boosting rules
    for category, rule in boosting_rules.items():
        for keyword in rule['keywords']:
            if keyword in result_lower:
                boosted_score += rule['boost']
    
    # Apply negative penalties
    for query_keyword, penalty_rule in negative_keywords.items():
        if query_keyword in query.lower():  # Only apply if query contains this keyword
            for avoid_keyword in penalty_rule['avoid']:
                if avoid_keyword in result_lower:
                    boosted_score -= penalty_rule['penalty']
    
    result_scores.append((max(0, boosted_score), found_query_words, similarity, result))

print("\n🔍 Most Relevant Results:")
print("=" * 80)

result_count = 0
for boosted_score, found_query_words, similarity, result in result_scores:
    # STRICT FILTER: If query has specific keywords, REQUIRE they be found in result
    critical_keywords = ['jyotirlinga', 'shakti', 'mutt', 'temple', 'statue', 'festival', 'deity', 'shrine']
    has_critical_keyword = any(keyword in query.lower() for keyword in critical_keywords)
    
    result_lower = result.lower()
    has_result_keyword = any(keyword in result_lower for keyword in critical_keywords)
    
    # If query mentions critical keywords, result MUST contain related keywords
    if has_critical_keyword and not has_result_keyword:
        continue  # Skip this result
    
    # Only show results that meet threshold AND found at least one query word
    if boosted_score > 0.40 and found_query_words > 0:
        result_count += 1
        truncated = result[:500] + "..." if len(result) > 500 else result
        print(f"\n📄 Result {result_count} (Relevance: {boosted_score:.2%}):")
        print(truncated)
        print("-" * 80)

if result_count == 0:
    print("\n❌ No highly relevant results found. Try a different query.")

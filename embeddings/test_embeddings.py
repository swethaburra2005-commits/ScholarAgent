from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Three example sentences
sentences = [
    "How can hallucinations in RAG systems be detected?",
    "Methods for detecting hallucinations in retrieval augmented generation.",
    "How do I cook pasta at home?"
]


# Convert the sentences into embeddings
embeddings = model.encode(sentences)


# Show the size of the embeddings
print("Embedding shape:")
print(embeddings.shape)


# Calculate semantic similarity
similarities = model.similarity(embeddings, embeddings)


print("\nSimilarity scores:")
print(similarities)
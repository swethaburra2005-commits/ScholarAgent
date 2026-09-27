from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def rank_papers(query, papers):

    # Create an embedding for the research question
    query_embedding = model.encode([query])

    # Create embeddings for all paper abstracts
    paper_abstracts = [
        paper["abstract"] for paper in papers
    ]

    paper_embeddings = model.encode(paper_abstracts)

    # Calculate similarity between query and every paper
    similarities = model.similarity(
        query_embedding,
        paper_embeddings
    )

    # Add similarity score to each paper
    for i, paper in enumerate(papers):
        paper["similarity_score"] = float(similarities[0][i])

    # Sort papers from most relevant to least relevant
    papers.sort(
        key=lambda paper: paper["similarity_score"],
        reverse=True
    )

    return papers 
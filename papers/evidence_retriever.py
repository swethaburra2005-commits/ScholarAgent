import json
import torch

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from sentence_transformers.util import semantic_search


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Load saved chunks and embeddings
# ---------------------------------------------------------

def load_embeddings(file_path):
    """
    Load chunks and their saved embeddings.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# ---------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------

def retrieve_candidates(
    query,
    data,
    top_k=10
):
    """
    Retrieve the top candidate chunks using
    Sentence Transformer embeddings.
    """

    print(
        "\nLoading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # Get saved document embeddings
    corpus_embeddings = [
        item["embedding"]
        for item in data
    ]

    corpus_embeddings = torch.tensor(
        corpus_embeddings
    )

    print(
        "\nEncoding research question..."
    )

    query_embedding = (
        embedding_model.encode_query(
            query,
            convert_to_tensor=True
        )
    )

    print(
        f"\nRetrieving top {top_k} candidates..."
    )

    results = semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=top_k
    )

    candidates = []

    for result in results[0]:

        chunk_index = result["corpus_id"]

        candidates.append(
            {
                "chunk_id": data[chunk_index][
                    "chunk_id"
                ],
                "text": data[chunk_index][
                    "text"
                ],
                "retrieval_score": float(
                    result["score"]
                )
            }
        )

    return candidates


# ---------------------------------------------------------
# Cross-Encoder reranking
# ---------------------------------------------------------

def rerank_candidates(
    query,
    candidates,
    top_k=5
):
    """
    Rerank retrieved candidates using
    a Cross-Encoder.
    """

    print(
        "\nLoading Cross-Encoder..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL,
        activation_fn=torch.nn.Sigmoid()
    )

    print(
        "Cross-Encoder loaded."
    )

    # Create query-document pairs
    pairs = []

    for candidate in candidates:

        pairs.append(
            [
                query,
                candidate["text"]
            ]
        )

    print(
        f"\nReranking {len(pairs)} candidates..."
    )

    scores = reranker.predict(
        pairs
    )

    reranked = []

    for candidate, score in zip(
        candidates,
        scores
    ):

        reranked.append(
            {
                "chunk_id": candidate[
                    "chunk_id"
                ],
                "text": candidate[
                    "text"
                ],
                "retrieval_score": candidate[
                    "retrieval_score"
                ],
                "rerank_score": float(
                    score
                )
            }
        )

    # Sort by Cross-Encoder score
    reranked.sort(
        key=lambda x: x[
            "rerank_score"
        ],
        reverse=True
    )

    return reranked[:top_k]


# ---------------------------------------------------------
# Display final evidence
# ---------------------------------------------------------

def display_results(results):
    """
    Display final reranked evidence.
    """

    print("\n" + "=" * 70)

    print(
        "SCHOLAR AGENT — RETRIEVE + RERANK"
    )

    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {rank}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Initial retrieval score: "
            f"{result['retrieval_score']:.4f}"
        )

        print(
            f"Cross-Encoder score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            "\nEvidence:"
        )

        print(
            result["text"]
        )

        print(
            "\n" + "-" * 70
        )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

if __name__ == "__main__":

    embeddings_file = (
        "papers/chunk_embeddings.json"
    )

    print(
        "\nLoading paper embeddings..."
    )

    data = load_embeddings(
        embeddings_file
    )

    print(
        "Chunks loaded:",
        len(data)
    )

    query = input(
        "\nAsk a question about the paper: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a question."
        )

        exit()

    # Stage 1:
    # Fast semantic retrieval
    candidates = retrieve_candidates(
        query,
        data,
        top_k=10
    )

    # Stage 2:
    # More precise Cross-Encoder reranking
    final_results = rerank_candidates(
        query,
        candidates,
        top_k=5
    )

    # Stage 3:
    # Display final evidence
    display_results(
        final_results
    )
import json

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_chunks(file_path):
    """
    Load the saved chunk embeddings file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def rerank_chunks(
    query,
    chunks,
    top_k=5
):
    """
    Rerank paper chunks using a Cross-Encoder.
    """

    print("\nLoading Cross-Encoder...")

    model = CrossEncoder(
        MODEL_NAME
    )

    print("Cross-Encoder loaded.")

    # Create query-document pairs
    pairs = []

    for chunk in chunks:
        pairs.append(
            (
                query,
                chunk["text"]
            )
        )

    print(
        f"\nReranking {len(pairs)} chunks..."
    )

    scores = model.predict(
        pairs
    )

    # Attach scores
    ranked_results = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        ranked_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score)
            }
        )

    # Sort by Cross-Encoder score
    ranked_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked_results[:top_k]


def display_results(results):
    """
    Display reranked evidence.
    """

    print("\n" + "=" * 70)
    print("SCHOLAR AGENT — CROSS-ENCODER RERANKING")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {rank}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Cross-Encoder score: "
            f"{result['score']:.4f}"
        )

        print("\nEvidence:")

        print(
            result["text"]
        )

        print("\n" + "-" * 70)


if __name__ == "__main__":

    embeddings_file = (
        "papers/chunk_embeddings.json"
    )

    print("\nLoading chunks...")

    chunks = load_chunks(
        embeddings_file
    )

    print(
        "Chunks loaded:",
        len(chunks)
    )

    query = input(
        "\nAsk a question about the paper: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a question."
        )

        exit()

    results = rerank_chunks(
        query,
        chunks,
        top_k=5
    )

    display_results(
        results
    )
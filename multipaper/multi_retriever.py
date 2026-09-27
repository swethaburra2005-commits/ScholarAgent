import json
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_FILE = (
    "papers/multi_paper_embeddings.json"
)


def load_embeddings():
    """
    Load the unified multi-paper embedding index.
    """

    print("\nLoading embedding index...")

    with open(
        EMBEDDING_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        f"Loaded {len(data)} embedded chunks."
    )

    return data


def retrieve_evidence(
    query,
    data,
    top_k=10
):
    """
    Retrieve the most relevant chunks
    from all papers.
    """

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------
    # Load corpus embeddings
    # --------------------------------------------------

    corpus_embeddings = torch.tensor(
        [
            item["embedding"]
            for item in data
        ]
    )

    # --------------------------------------------------
    # Encode user question
    # --------------------------------------------------

    print(
        "\nEncoding research question..."
    )

    query_embedding = model.encode_query(
        query,
        convert_to_tensor=True
    )

    # --------------------------------------------------
    # Semantic search
    # --------------------------------------------------

    print(
        f"\nSearching across "
        f"{len(data)} chunks..."
    )

    results = semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=top_k
    )

    return results[0]


def display_results(
    results,
    data
):
    """
    Display retrieved evidence
    with source information.
    """

    print(
        "\n" + "=" * 80
    )

    print(
        "SCHOLAR AGENT — MULTI-PAPER EVIDENCE RETRIEVAL"
    )

    print(
        "=" * 80
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        index = result[
            "corpus_id"
        ]

        score = result[
            "score"
        ]

        item = data[index]

        print(
            f"\nRESULT {rank}"
        )

        print(
            "-" * 80
        )

        print(
            f"Paper ID: "
            f"{item['paper_id']}"
        )

        print(
            f"Title: "
            f"{item['title']}"
        )

        print(
            f"Chunk ID: "
            f"{item['chunk_id']}"
        )

        print(
            f"Similarity score: "
            f"{score:.4f}"
        )

        print(
            f"\nSource:"
        )

        print(
            item["arxiv_url"]
        )

        print(
            "\nEvidence:"
        )

        print(
            item["text"]
        )

        print(
            "\n" + "-" * 80
        )


if __name__ == "__main__":

    data = load_embeddings()

    query = input(
        "\nEnter your research question: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a research question."
        )

        exit()

    results = retrieve_evidence(
        query,
        data,
        top_k=10
    )

    display_results(
        results,
        data
    )
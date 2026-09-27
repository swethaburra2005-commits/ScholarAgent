import json
import torch

from functools import lru_cache

from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer
)

from sentence_transformers.util import semantic_search


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

EMBEDDING_FILE = (
    "papers/multi_paper_embeddings.json"
)


# ============================================================
# CACHED MODELS
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    return model


@lru_cache(maxsize=1)
def get_reranker_model():

    print("\nLoading Cross-Encoder...")

    model = CrossEncoder(
        RERANKER_MODEL,
        activation_fn=torch.nn.Sigmoid()
    )

    print("Cross-Encoder loaded.")

    return model


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print("\nLoading embedding index...")

    with open(
        EMBEDDING_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        f"Loaded {len(data)} chunks."
    )

    return data


# ============================================================
# FIRST-STAGE RETRIEVAL
# ============================================================

def retrieve_candidates(
    query,
    data,
    top_k=10
):

    model = get_embedding_model()

    corpus_embeddings = torch.tensor(
        [
            item["embedding"]
            for item in data
        ]
    )

    print(
        "\nEncoding question..."
    )

    query_embedding = model.encode_query(
        query,
        convert_to_tensor=True
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

        index = result["corpus_id"]

        item = data[index]

        candidates.append(
            {
                "paper_id": item["paper_id"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "arxiv_url": item["arxiv_url"],
                "text": item["text"],
                "retrieval_score": float(
                    result["score"]
                )
            }
        )

    return candidates


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_candidates(
    query,
    candidates,
    top_k=5
):

    reranker = get_reranker_model()

    pairs = []

    for candidate in candidates:

        pairs.append(
            (
                query,
                candidate["text"]
            )
        )

    print(
        f"\nReranking {len(pairs)} candidates..."
    )

    scores = reranker.predict(
        pairs
    )

    results = []

    for candidate, score in zip(
        candidates,
        scores
    ):

        results.append(
            {
                **candidate,
                "rerank_score": float(
                    score
                )
            }
        )

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print(
        "\n" + "=" * 80
    )

    print(
        "SCHOLAR AGENT"
    )

    print(
        "MULTI-PAPER RETRIEVE + RERANK"
    )

    print(
        "=" * 80
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {rank}"
        )

        print(
            "-" * 80
        )

        print(
            f"Paper ID: "
            f"{result['paper_id']}"
        )

        print(
            f"Title: "
            f"{result['title']}"
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
            "\nSource:"
        )

        print(
            result["arxiv_url"]
        )

        print(
            "\nEvidence:"
        )

        print(
            result["text"]
        )

        print(
            "\n" + "-" * 80
        )


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    data = load_embeddings()

    query = input(
        "\nEnter your research question: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a question."
        )

        exit()

    candidates = retrieve_candidates(
        query,
        data,
        top_k=10
    )

    final_results = rerank_candidates(
        query,
        candidates,
        top_k=5
    )

    display_results(
        final_results
    )
import json
import os

from sentence_transformers import SentenceTransformer


CORPUS_FOLDER = "papers/corpus"

OUTPUT_FILE = "papers/multi_paper_embeddings.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_all_chunks():
    """
    Load chunks from all papers.
    """

    all_chunks = []

    for name in sorted(
        os.listdir(CORPUS_FOLDER)
    ):

        paper_folder = os.path.join(
            CORPUS_FOLDER,
            name
        )

        if not (
            os.path.isdir(paper_folder)
            and name.startswith("paper_")
        ):
            continue

        chunks_file = os.path.join(
            paper_folder,
            "chunks.json"
        )

        if not os.path.exists(
            chunks_file
        ):
            print(
                f"Skipping {name}: "
                "chunks.json not found."
            )

            continue

        with open(
            chunks_file,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        print(
            f"{name}: "
            f"{len(chunks)} chunks"
        )

        all_chunks.extend(chunks)

    return all_chunks


def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "SCHOLAR AGENT — MULTI-PAPER EMBEDDING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------
    # Load chunks
    # --------------------------------------------------

    print(
        "\nLoading chunks..."
    )

    chunks = load_all_chunks()

    print(
        f"\nTotal chunks loaded: "
        f"{len(chunks)}"
    )

    if not chunks:

        print(
            "\nNo chunks found."
        )

        return

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------
    # Prepare text
    # --------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------
    # Create embeddings
    # --------------------------------------------------

    print(
        "\nCreating embeddings..."
    )

    embeddings = model.encode_document(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(
        "\nEmbedding complete."
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------
    # Attach embeddings to metadata
    # --------------------------------------------------

    records = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        record = {
            "paper_id": chunk[
                "paper_id"
            ],
            "chunk_id": chunk[
                "chunk_id"
            ],
            "title": chunk[
                "title"
            ],
            "arxiv_url": chunk[
                "arxiv_url"
            ],
            "pdf_url": chunk[
                "pdf_url"
            ],
            "text": chunk[
                "text"
            ],
            "embedding": embedding.tolist()
        }

        records.append(
            record
        )

    # --------------------------------------------------
    # Save unified index
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file
        )

    print(
        f"\nSaved unified embedding index:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nRecords saved:",
        len(records)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTI-PAPER EMBEDDING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main() 
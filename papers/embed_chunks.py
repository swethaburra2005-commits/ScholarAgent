import json
import re

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(file_path):
    """
    Load chunks from test_chunks.txt.

    Looks for sections such as:

    CHUNK 1
    --------
    text...

    CHUNK 2
    --------
    text...
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        content = file.read()

    # Find every CHUNK heading and everything
    # until the next CHUNK heading.
    pattern = r"CHUNK\s+\d+\s*(.*?)(?=CHUNK\s+\d+|$)"

    matches = re.findall(
        pattern,
        content,
        flags=re.DOTALL
    )

    chunks = []

    for match in matches:

        chunk = match.strip()

        # Remove separator lines consisting only of "="
        chunk = re.sub(
            r"^=+\s*",
            "",
            chunk
        )

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def create_embeddings(chunks):
    """
    Convert every paper chunk into an embedding.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Model loaded.")

    print(
        f"\nCreating embeddings for "
        f"{len(chunks)} chunks..."
    )

    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    print("\nEmbedding creation complete.")

    print(
        "Embedding shape:",
        embeddings.shape
    )

    return embeddings


def save_embeddings(
    chunks,
    embeddings,
    output_file
):
    """
    Save chunks and their embeddings
    into a JSON file.
    """

    data = []

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings),
        start=1
    ):

        data.append(
            {
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding.tolist()
            }
        )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )

    print(
        "\nEmbeddings saved successfully!"
    )

    print(
        "Saved to:",
        output_file
    )


if __name__ == "__main__":

    input_file = "papers/test_chunks.txt"

    output_file = (
        "papers/chunk_embeddings.json"
    )

    print("\nLoading chunks...")

    chunks = load_chunks(
        input_file
    )

    print(
        "Chunks loaded:",
        len(chunks)
    )

    if not chunks:

        print(
            "\nERROR: No chunks were found."
        )

        print(
            "Please check:",
            input_file
        )

        exit()

    # Show a preview so we know the
    # chunks were loaded correctly.
    print("\nFirst chunk preview:")
    print("-" * 70)
    print(chunks[0][:500])
    print("-" * 70)

    embeddings = create_embeddings(
        chunks
    )

    save_embeddings(
        chunks,
        embeddings,
        output_file
    )

    print("\n" + "=" * 70)
    print("EMBEDDING STAGE COMPLETE")
    print("=" * 70)
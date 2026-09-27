def load_text(file_path):
    """
    Load extracted paper text.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


def split_into_sentences(text):
    """
    Simple sentence splitting.
    """

    sentences = []

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Basic sentence splitting
        current = ""

        for character in paragraph:

            current += character

            if character in ".?!":

                if current.strip():
                    sentences.append(
                        current.strip()
                    )

                current = ""

        if current.strip():
            sentences.append(
                current.strip()
            )

    return sentences


def create_chunks(
    text,
    chunk_size=1500,
    overlap=200
):
    """
    Create chunks while trying to preserve
    sentence boundaries.
    """

    sentences = split_into_sentences(text)

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        # If adding this sentence stays within
        # the chunk size, keep adding it.
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:

            if current_chunk:
                current_chunk += " "

            current_chunk += sentence

        else:

            # Save current chunk
            if current_chunk.strip():
                chunks.append(
                    current_chunk.strip()
                )

            # Keep some overlap from the
            # end of the previous chunk
            overlap_text = current_chunk[-overlap:]

            current_chunk = (
                overlap_text
                + " "
                + sentence
            )

            # If one sentence itself is too long,
            # hard-split it.
            if len(current_chunk) > chunk_size:

                chunks.append(
                    current_chunk[:chunk_size].strip()
                )

                current_chunk = (
                    current_chunk[
                        chunk_size - overlap:
                    ]
                )

    # Add final chunk
    if current_chunk.strip():
        chunks.append(
            current_chunk.strip()
        )

    return chunks


def save_chunks(chunks, output_file):
    """
    Save chunks to a text file.
    """

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            file.write(
                "\n"
                + "=" * 70
                + "\n"
            )

            file.write(
                f"CHUNK {i}\n"
            )

            file.write(
                "=" * 70
                + "\n\n"
            )

            file.write(chunk)

            file.write("\n\n")


if __name__ == "__main__":

    input_file = "papers/test_paper.txt"
    output_file = "papers/test_chunks.txt"

    print("\nLoading paper text...")

    text = load_text(input_file)

    print(
        "Total characters:",
        len(text)
    )

    print("\nCreating chunks...")

    chunks = create_chunks(text)

    print(
        "Number of chunks:",
        len(chunks)
    )

    save_chunks(
        chunks,
        output_file
    )

    print(
        "\nChunks saved successfully!"
    )

    print(
        "Saved to:",
        output_file
    )

    print("\nFirst chunk:")
    print("=" * 70)
    print(chunks[0])

    print("\n\nLast chunk:")
    print("=" * 70)
    print(chunks[-1])
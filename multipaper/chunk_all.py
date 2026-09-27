import json
import os
import re


CORPUS_FOLDER = "papers/corpus"

CHUNK_SIZE = 1500
OVERLAP = 200


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
    Split text approximately into sentences.
    """

    sentences = []

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Clean excessive whitespace
        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph
        )

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
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP
):
    """
    Create overlapping chunks while
    attempting to preserve sentence boundaries.
    """

    sentences = split_into_sentences(
        text
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if (
            len(current_chunk)
            + len(sentence)
            + 1
            <= chunk_size
        ):

            if current_chunk:

                current_chunk += " "

            current_chunk += sentence

        else:

            if current_chunk.strip():

                chunks.append(
                    current_chunk.strip()
                )

            overlap_text = (
                current_chunk[-overlap:]
            )

            current_chunk = (
                overlap_text
                + " "
                + sentence
            )

            # Handle very long sentences
            if len(current_chunk) > chunk_size:

                chunks.append(
                    current_chunk[
                        :chunk_size
                    ].strip()
                )

                current_chunk = (
                    current_chunk[
                        chunk_size - overlap:
                    ]
                )

    if current_chunk.strip():

        chunks.append(
            current_chunk.strip()
        )

    return chunks


def load_metadata(metadata_path):

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def process_paper(paper_folder):

    paper_id = os.path.basename(
        paper_folder
    )

    text_path = os.path.join(
        paper_folder,
        "paper.txt"
    )

    metadata_path = os.path.join(
        paper_folder,
        "metadata.json"
    )

    output_path = os.path.join(
        paper_folder,
        "chunks.json"
    )

    if not os.path.exists(text_path):

        print(
            "Text file not found."
        )

        return 0

    print(
        f"\nProcessing {paper_id}..."
    )

    text = load_text(
        text_path
    )

    metadata = load_metadata(
        metadata_path
    )

    chunks = create_chunks(
        text
    )

    chunk_records = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_records.append(
            {
                "paper_id": paper_id,
                "chunk_id": i,
                "title": metadata.get(
                    "title",
                    ""
                ),
                "arxiv_url": metadata.get(
                    "url",
                    ""
                ),
                "pdf_url": metadata.get(
                    "pdf_url",
                    ""
                ),
                "text": chunk
            }
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunk_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"   Characters: {len(text)}"
    )

    print(
        f"   Chunks created: "
        f"{len(chunks)}"
    )

    print(
        f"   Saved: {output_path}"
    )

    return len(chunks)


if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "SCHOLAR AGENT — MULTI-PAPER CHUNKING"
    )

    print(
        "=" * 70
    )

    if not os.path.exists(
        CORPUS_FOLDER
    ):

        print(
            "\nCorpus folder not found."
        )

        exit()

    paper_folders = []

    for name in os.listdir(
        CORPUS_FOLDER
    ):

        folder_path = os.path.join(
            CORPUS_FOLDER,
            name
        )

        if (
            os.path.isdir(folder_path)
            and name.startswith("paper_")
        ):

            paper_folders.append(
                folder_path
            )

    paper_folders.sort()

    print(
        f"\nFound {len(paper_folders)} papers."
    )

    total_chunks = 0

    for paper_folder in paper_folders:

        total_chunks += process_paper(
            paper_folder
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTI-PAPER CHUNKING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal chunks created: "
        f"{total_chunks}"
    )
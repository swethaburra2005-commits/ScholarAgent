import os

from pypdf import PdfReader


CORPUS_FOLDER = "papers/corpus"


def extract_pdf(pdf_path):
    """
    Extract text from one PDF.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        print(
            f"      Page "
            f"{page_number}/{len(reader.pages)}"
        )

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def process_paper(paper_folder):
    """
    Extract one paper and save its text.
    """

    pdf_path = os.path.join(
        paper_folder,
        "paper.pdf"
    )

    text_path = os.path.join(
        paper_folder,
        "paper.txt"
    )

    if not os.path.exists(pdf_path):

        print(
            "      PDF not found."
        )

        return False

    print(
        f"\n   Reading: {pdf_path}"
    )

    try:

        text = extract_pdf(
            pdf_path
        )

        if not text.strip():

            print(
                "      No text extracted."
            )

            return False

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

        print(
            f"      Saved: {text_path}"
        )

        print(
            f"      Characters: {len(text)}"
        )

        return True

    except Exception as error:

        print(
            f"      Extraction failed: {error}"
        )

        return False


if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "SCHOLAR AGENT — MULTI-PAPER EXTRACTION"
    )

    print(
        "=" * 70
    )

    if not os.path.exists(
        CORPUS_FOLDER
    ):

        print(
            "\nCorpus folder not found:"
        )

        print(
            CORPUS_FOLDER
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

    successful = 0

    for i, paper_folder in enumerate(
        paper_folders,
        start=1
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"PAPER {i}"
        )

        print(
            "-" * 70
        )

        success = process_paper(
            paper_folder
        )

        if success:
            successful += 1

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTI-PAPER EXTRACTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSuccessfully extracted: "
        f"{successful}/{len(paper_folders)}"
    )
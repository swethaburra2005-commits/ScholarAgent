from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns the complete extracted text.
    """

    reader = PdfReader(pdf_path)

    print("\nPDF loaded successfully.")
    print("Number of pages:", len(reader.pages))

    all_text = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        print(
            f"Extracting page "
            f"{page_number}/{len(reader.pages)}..."
        )

        text = page.extract_text()

        if text:
            all_text.append(text)

    complete_text = "\n\n".join(all_text)

    return complete_text


def save_text(text, output_path):
    """
    Save extracted text to a UTF-8 text file.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)

    print("\nText saved successfully!")
    print("Saved to:", output_path)


if __name__ == "__main__":

    pdf_path = input(
        "\nEnter PDF path: "
    ).strip()

    text = extract_text_from_pdf(pdf_path)

    if text:

        print("\n" + "=" * 70)
        print("TEXT EXTRACTION SUCCESSFUL")
        print("=" * 70)

        print("\nTotal characters extracted:")
        print(len(text))

        # Save extracted text
        output_path = "papers/test_paper.txt"

        save_text(
            text,
            output_path
        )

    else:

        print(
            "\nNo text could be extracted "
            "from this PDF."
        )
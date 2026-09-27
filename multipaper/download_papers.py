import json
import os
import time

import requests

from search_papers import search_arxiv


DOWNLOAD_FOLDER = "papers/corpus"


def download_pdf(
    pdf_url,
    output_path
):
    """
    Download one PDF from arXiv.
    """

    print(
        f"\nDownloading:\n{pdf_url}"
    )

    response = requests.get(
        pdf_url,
        timeout=60
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if "application/pdf" not in content_type:
        raise ValueError(
            "Downloaded content is not a PDF."
        )

    with open(
        output_path,
        "wb"
    ) as file:

        file.write(
            response.content
        )

    print(
        f"Saved: {output_path}"
    )


def save_metadata(
    paper,
    output_path
):
    """
    Save paper metadata as JSON.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            paper,
            file,
            indent=2,
            ensure_ascii=False
        )


if __name__ == "__main__":

    query = input(
        "\nEnter a research topic: "
    ).strip()

    if not query:

        print(
            "Please enter a research topic."
        )

        exit()

    print(
        "\nSearching for papers..."
    )

    papers = search_arxiv(
        query,
        max_results=5
    )

    print(
        f"\nFound {len(papers)} papers."
    )

    os.makedirs(
        DOWNLOAD_FOLDER,
        exist_ok=True
    )

    for i, paper in enumerate(
        papers,
        start=1
    ):

        paper_folder = os.path.join(
            DOWNLOAD_FOLDER,
            f"paper_{i:03d}"
        )

        os.makedirs(
            paper_folder,
            exist_ok=True
        )

        pdf_path = os.path.join(
            paper_folder,
            "paper.pdf"
        )

        metadata_path = os.path.join(
            paper_folder,
            "metadata.json"
        )

        print(
            "\n" + "=" * 70
        )

        print(
            f"PAPER {i}"
        )

        print(
            f"Title: {paper['title']}"
        )

        try:

            download_pdf(
                paper["pdf_url"],
                pdf_path
            )

            save_metadata(
                paper,
                metadata_path
            )

            print(
                "Download successful!"
            )

        except Exception as error:

            print(
                f"Download failed: {error}"
            )

        # Be polite to the API/server.
        time.sleep(3)

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTI-PAPER DOWNLOAD COMPLETE"
    )

    print(
        "=" * 70
    )
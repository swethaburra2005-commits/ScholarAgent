import requests
import feedparser
from urllib.parse import quote_plus


ARXIV_API_URL = (
    "https://export.arxiv.org/api/query"
)


def search_arxiv(
    query,
    max_results=5
):
    """
    Search arXiv for research papers.
    """

    encoded_query = quote_plus(
        f"all:{query}"
    )

    url = (
        f"{ARXIV_API_URL}"
        f"?search_query={encoded_query}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=relevance"
        f"&sortOrder=descending"
    )

    print(
        "\nSearching arXiv..."
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    feed = feedparser.parse(
        response.text
    )

    papers = []

    for entry in feed.entries:

        paper = {
            "title": entry.title.strip(),
            "abstract": entry.summary.strip(),
            "url": entry.id,
            "pdf_url": None
        }

        # Find PDF link
        for link in entry.links:

            if link.get("type") == "application/pdf":

                paper["pdf_url"] = (
                    link.href
                )

                break

        papers.append(
            paper
        )

    return papers


if __name__ == "__main__":

    query = input(
        "\nEnter a research topic: "
    ).strip()

    if not query:

        print(
            "Please enter a research topic."
        )

        exit()

    papers = search_arxiv(
        query,
        max_results=5
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ARXIV SEARCH RESULTS"
    )

    print(
        "=" * 70
    )

    for i, paper in enumerate(
        papers,
        start=1
    ):

        print(
            f"\nPAPER {i}"
        )

        print(
            f"Title: {paper['title']}"
        )

        print(
            f"URL: {paper['url']}"
        )

        print(
            f"PDF: {paper['pdf_url']}"
        )

        print(
            "\nAbstract:"
        )

        print(
            paper["abstract"][:500]
        )

        print(
            "\n" + "-" * 70
        )
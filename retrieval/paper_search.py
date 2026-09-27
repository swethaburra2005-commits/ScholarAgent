import requests
import xml.etree.ElementTree as ET

from embeddings.paper_ranker import rank_papers


def search_papers(query, max_results=5):
    """
    Search arXiv for academic papers.

    Returns a list of dictionaries containing:
    - title
    - authors
    - published date
    - abstract
    - paper URL
    - PDF URL
    """

    url = "http://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error contacting arXiv.")
        print("Status code:", response.status_code)
        return []

    root = ET.fromstring(response.text)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):

       

        title = entry.find(
            "atom:title",
            namespace
        ).text.strip()

        abstract = entry.find(
            "atom:summary",
            namespace
        ).text.strip()

        published = entry.find(
            "atom:published",
            namespace
        ).text.strip()

        paper_url = entry.find(
            "atom:id",
            namespace
        ).text.strip()

       

        pdf_url = None

        for link in entry.findall(
            "atom:link",
            namespace
        ):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

       
        authors = []

        for author in entry.findall(
            "atom:author",
            namespace
        ):
            name = author.find(
                "atom:name",
                namespace
            ).text

            authors.append(name)

        
        paper = {
            "title": title,
            "authors": authors,
            "published": published,
            "abstract": abstract,
            "url": paper_url,
            "pdf_url": pdf_url
        }

        papers.append(paper)

    return papers


def display_papers(papers, query):
    """
    Display ranked papers in a clean format.
    """

    print("\n" + "=" * 70)
    print("SCHOLAR AGENT — PAPER SEARCH")
    print("=" * 70)

    print(f"\nResearch question: {query}")
    print(f"Papers found: {len(papers)}")

    for i, paper in enumerate(papers, start=1):

        print("\n" + "-" * 70)

        print(f"\n{i}. {paper['title']}")

        
        if "similarity_score" in paper:
            print(
                f"Relevance score: "
                f"{paper['similarity_score']:.4f}"
            )

        # Authors
        print("\nAuthors:")
        print(", ".join(paper["authors"]))

        # Publication date
        print("\nPublished:")
        print(paper["published"])

        # Abstract
        print("\nAbstract:")
        print(paper["abstract"])

        # Abstract page
        print("\nPaper URL:")
        print(paper["url"])

        # PDF
        print("\nPDF:")
        print(paper["pdf_url"])


def main():
    """
    Main ScholarAgent paper-search workflow.
    """

    query = input(
        "\nEnter your research question: "
    ).strip()

    if not query:
        print("Please enter a research question.")
        return

    papers = search_papers(query)

    if not papers:
        print("\nNo papers were found.")
        return

  
    papers = rank_papers(query, papers)

  
    display_papers(papers, query)


if __name__ == "__main__":
    main() 
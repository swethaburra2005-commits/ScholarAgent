import os
import requests


def download_pdf(pdf_url, filename):
    """
    Download a PDF from a URL and verify that
    the downloaded file looks like a PDF.
    """

    os.makedirs("papers", exist_ok=True)

    file_path = os.path.join("papers", filename)

    print("\nDownloading paper...")
    print("URL:", pdf_url)

    try:
        response = requests.get(
            pdf_url,
            timeout=60
        )

        response.raise_for_status()

    except requests.RequestException as error:
        print("\nDownload failed.")
        print("Error:", error)
        return None

    # Check that the response is actually a PDF
    content_type = response.headers.get(
        "Content-Type",
        ""
    )

    print("\nContent-Type:")
    print(content_type)

    if "pdf" not in content_type.lower():
        print("\nWarning: response does not look like a PDF.")

    # Get downloaded data
    pdf_data = response.content

    print("\nDownloaded size:")
    print(f"{len(pdf_data):,} bytes")

    # A real PDF should begin with %PDF
    if not pdf_data.startswith(b"%PDF"):
        print("\nError: downloaded file is not a valid PDF.")
        print("The server may have returned HTML or another response.")
        return None

    # Save PDF
    with open(file_path, "wb") as file:
        file.write(pdf_data)

    print("\nDownload successful!")
    print("Saved to:", file_path)

    # Verify the file exists and has a reasonable size
    file_size = os.path.getsize(file_path)

    print("File size:")
    print(f"{file_size:,} bytes")

    return file_path


if __name__ == "__main__":

    pdf_url = input(
        "Enter the PDF URL: "
    ).strip()

    filename = "test_paper.pdf"

    download_pdf(
        pdf_url,
        filename
    )
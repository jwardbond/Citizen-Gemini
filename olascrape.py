import json
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Base URL for the Hansard documents page
BILL_BASE_URL = (
    "https://www.ola.org/en/legislative-business/bills/parliament-43/session-1/"
)
HANS_BASE_URL = "https://www.ola.org/en/legislative-business/house-documents/parliament-43/session-1/"
DATE_PATTERN = r"(\d{4}-\d{2}-\d{2})"


#
# BILL SCRAPING
#
def get_bill_names() -> list[str]:
    response = requests.get(BILL_BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    elements = soup.find_all("td", class_="views-field views-field-field-bill-number")
    bills = [f"bill-{x.get_text(strip=True)}" for x in elements]

    return bills


def fetch_bill_contents(bill: str) -> str:
    url = BILL_BASE_URL + bill

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    status = (
        soup.find("p", class_=re.compile("views-field-field-current-status*"))
        .get_text(strip=True)
        .replace("\n", "")
        .replace("\t", "")
        .replace("   ", " ")
    )
    try:
        if "out of order" in status.lower():
            sponsor = "Withdrawn"
            text = soup.findAll("p")[1].get_text().replace("\r\n", " ")
            text = "\n\n" + text
        else:
            sponsor = (
                soup.find("div", class_="views-field-field-member")
                .get_text()
                .replace("\n", "")
            )
            text = (
                soup.find("div", class_="WordSection1").get_text().replace("\r\n", " ")
            )
        return f"sponsor: {sponsor}\n{status}{text}"
    except AttributeError as e:
        print(url, "\n", e)
        return "None"


#
# HANSARD SCRAPING
#


def get_hansard_dates() -> list[str]:
    response = requests.get(HANS_BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    dates = {
        re.search(DATE_PATTERN, urljoin(HANS_BASE_URL, link["href"])).group(0)
        for link in soup.find_all("a", href=True)
        if re.search(DATE_PATTERN, link["href"])
    }
    return list(dates)


def fetch_hansard_content(date: str) -> str:
    # Links are like hansard, then hansard-1, hansard-2...
    combined_content = ""
    page_num = (
        0  # Using a separate variable to avoid overwriting the outer loop's index
    )

    while True:
        suffix = f"-{page_num}" if page_num > 0 else ""
        url = f"{HANS_BASE_URL}{date}/hansard{suffix}"

        response = requests.get(url)
        if response.status_code != 200:
            break  # Stop if the page doesn't exist

        # Parse and extract main content
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("div", {"id": "transcript"})

        if main_content:
            combined_content += main_content.get_text(separator="\n")
        else:
            print(f"Warning: Content not found on {url}")

        page_num += 1  # Move to the next page with suffix

    combined_content = re.sub(r"\\x[0-9a-fA-F]{2}", "", combined_content)
    combined_content = re.sub(r"\n\n", "\n", combined_content)
    combined_content = re.sub(r"\n{4,}", "\n\n\n", combined_content)
    combined_content = re.sub(r":\n", ":", combined_content)

    return combined_content


def scrape() -> None:
    """Scrapes bill and hansard (transcript) data from Ontario Legislature.

    Creates a documents.json with the following structure:

        ```
        {
            document_id: {
                type: "typename"
                text: "document contents"
            }
        }
        ```

    In this case, `document_id` is "bill-[bill number]" for bills, and "tr-yyyy-mm-dd" for transcripts
    """
    bills = get_bill_names()
    bill_data = {b: {"type": "bill", "contents": fetch_bill_contents(b)} for b in bills}

    dates = get_hansard_dates()
    hans_data = {
        d: {"type": "transcript", "text": fetch_hansard_content(d)} for d in dates
    }

    data = dict(bill_data, **hans_data)

    with open("documents.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    scrape()

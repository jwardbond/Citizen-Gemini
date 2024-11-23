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
def get_bill_names() -> list[tuple]:
    """Returns bill ids and titles for all bills in current legislature."""
    response = requests.get(BILL_BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    elements = soup.find_all("td", class_="views-field views-field-field-bill-number")
    bills = [f"bill-{x.get_text(strip=True)}" for x in elements]  # pr bills are useless

    elements = soup.find_all("td", class_="views-field views-field-field-short-title")
    titles = [f"{x.get_text(strip=True)}" for x in elements]

    output = zip(bills, titles, strict=True)
    output = [(x[0], x[1]) for x in output if "pr" not in x[0].lower()]

    return output


def fetch_bill_contents(bill: tuple[str]) -> str:
    """Gets and formats bill contents.

    Args:
        bill (tuple): a tuple containing (id, title)
    """

    bill_id = bill[0]
    bill_title = bill[1]

    url = BILL_BASE_URL + bill_id

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

        # Get rid of any special hex characters
        text = text.replace("\xa0", " ")
        text = re.sub(r"\\x[0-9A-Fa-f]{2}", "", text)

        return f"Title: {bill_title}\nSponsor: {sponsor}\n{status}{text}"
    except AttributeError as e:
        print(url, "\n", e)
        return "None"


#
# HANSARD SCRAPING
#
def get_hansard_dates() -> list[str]:
    """Returns a list of yyyy-mm-dd date strings."""
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


#
# DOCUMENT SUMMARIZING
#
def generate_transcript_summary(id: str, text: str) -> str:
    """Process and store topics and speakers for each transcript."""
    date = id.replace("tr-", "")

    # Find Topics, skipping header lines and empty lines
    toc_lines = text.split("\n\n")[0:50]
    topics = [
        line.strip()
        for line in toc_lines
        if line.strip()
        and "LEGISLATIVE ASSEMBLY" not in line
        and "ASSEMBLÉE LÉGISLATIVE" not in line
        and not line.startswith("Thursday")
        and not line.startswith("Monday")
        and not line.startswith("Tuesday")
        and not line.startswith("Wednesday")
        and not line.startswith("Friday")
        and not line.startswith("Jeudi")
        and not line.startswith("Lundi")
        and not line.startswith("Mardi")
        and not line.startswith("Mercredi")
        and not line.startswith("Vendredi")
    ]

    # Find speakers in Question Period and other sections
    # Look for patterns like:
    # "Mr. Smith:", "Ms. Jones:", "Hon. Doug Ford:", "The Speaker:"
    # This check is imperfect, but it's a good enough heuristic
    # And we will toss it into LLM anyways
    """
    Example of failure mode (has colon and starts with The):
    The 911 model of care that we referenced at the Association of Municipalities of Ontario conference earlier this week has been embraced: community paramedicine that allows community paramedics to go into those homes, for individuals who are able, in most cases with very little support, to stay safely in their home. The municipalities that have embraced that 911 model of care have loved it. In fact, our satisfaction rate, I believe, is in the 97th percentile.
    """
    speakers = set()
    lines = text.split("\n")
    for line in lines:
        # Process lines that start with a title prefix
        if (
            any(
                line.strip().startswith(prefix)
                for prefix in ["Mr.", "Ms.", "Mrs.", "Hon.", "The"]
            )
            and ":" in line
        ):
            speaker = line.split(":")[0].strip()
            if ("(" in speaker and ")" in speaker) or any(
                title in speaker for title in ["Mr.", "Ms.", "Mrs.", "Hon."]
            ):
                speakers.add(speaker)

    speakers = list(speakers)

    # Find bills
    # Bills have a simple pattern: "Bill 123A"
    bills = set()
    bill_pattern = re.compile(r"Bill\s+\d+[A-Za-z]*")

    for line in lines:
        bill_matches = bill_pattern.findall(line)
        bills.update(bill_matches)

    bills = list(bills)

    transcript_summary = f"{id} -- Transcript from: {date} | Speakers: {', '.join(speakers)} | Topics: {', '.join(topics)} | Bills: {', '.join(bills) if bills else 'None'}"

    return transcript_summary


def generate_bill_summary(id: str, text: str) -> str:
    bill_num = id.replace("bill-", "")

    pattern = rf"^(.*?)(?=Bill {bill_num}\s+\d{{4}}\s*An Act)"
    match = re.search(pattern, text, re.DOTALL)

    result = match.group(1) if match else text

    result = (
        result.replace("\n", " ")
        .replace(".", ". ")
        .replace("Sponsor:", " | Sponsor:")
        .replace("Current status:", " | Current status:")
        .replace("  ", " ")
    )
    result = re.sub(
        "EXPLANATORY NOTE",
        " | Explanatory Note: ",
        result,
        flags=re.IGNORECASE,
        count=1,
    )

    result = result.strip()

    return f"{id} -- Bill {bill_num} | {result}"


def summarize(docs: dict) -> None:
    summaries = {}
    for k, v in docs.items():
        if "tr" in k:
            summaries[k] = generate_transcript_summary(k, v["text"])
        elif "bill" in k:
            summaries[k] = generate_bill_summary(k, v["text"])

    return summaries


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
    bill_data = {b[0]: {"type": "bill", "text": fetch_bill_contents(b)} for b in bills}

    dates = get_hansard_dates()
    hans_data = {
        f"tr-{d}": {"type": "transcript", "text": fetch_hansard_content(d)}
        for d in dates
    }

    documents = dict(bill_data, **hans_data)

    return documents


if __name__ == "__main__":
    documents = scrape()

    with open("documents.json", "w") as f:
        json.dump(documents, f)

    summaries = summarize(documents)

    with open("summaries.json", "w") as f:
        json.dump(summaries, f)

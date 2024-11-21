import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import json

# Base URL for the Hansard documents page
BASE_URL = "https://www.ola.org/en/legislative-business/house-documents/parliament-43/session-1/"
DATE_PATTERN = r"(\d{4}-\d{2}-\d{2})"

def get_hansard_dates():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    dates = {
        re.search(DATE_PATTERN, urljoin(BASE_URL, link["href"])).group(0)
        for link in soup.find_all("a", href=True)
        if re.search(DATE_PATTERN, link["href"])
    }
    return list(dates)


def fetch_hansard_content(dates):
    content = {}
    for idx, date in enumerate(dates):
        print(f"Fetching date {date} ({idx + 1} / {len(dates)})")
        # Links are like hansard, then hansard-1, hansard-2...
        combined_content = ""
        page_num = 0  # Using a separate variable to avoid overwriting the outer loop's index

        while True:
            suffix = f"-{page_num}" if page_num > 0 else ""
            url = f"{BASE_URL}{date}/hansard{suffix}"

            print(f"\tFetching {url}")

            response = requests.get(url)
            if response.status_code != 200:
                print("\tNot found!")
                break  # Stop if the page doesn't exist

            # Parse and extract main content
            soup = BeautifulSoup(response.text, "html.parser")
            main_content = soup.find("div",{"id": "transcript"})

            if main_content:
                combined_content += main_content.get_text(separator="\n")
            else:
                print(f"Warning: Content not found on {url}")

            page_num += 1  # Move to the next page with suffix

        # Store the combined content for the specific date
        if combined_content:
            content[date] = combined_content

        print("\tDone scraping this date")

    return content


dates = get_hansard_dates()

print(f"Fetched dates (n={len(dates)})\n\n{dates}")

hansard = fetch_hansard_content(dates)

with open("hansard.json", "w", encoding="utf-8") as f:
    json.dump(hansard, f, ensure_ascii=False, indent=4)
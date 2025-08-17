\
import argparse
import json
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

def scrape_job_descriptions(url: str, num_pages: int = 1, delay: float = 0.5):
    """
    Scrape job postings from a site.
    Defaults to the Real Python Fake Jobs demo page structure.
    Falls back gracefully if selectors differ.
    """
    items = []
    for page in range(1, num_pages + 1):
        page_url = url
        # If the site uses pagination with ?page=, uncomment the next line:
        # page_url = f"{url}?page={page}"
        try:
            resp = requests.get(page_url, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                print(f"[WARN] HTTP {resp.status_code} for {page_url}")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # Try common patterns (Real Python Fake Jobs style)
            cards = soup.select("div.card-content")
            if not cards:
                # Fallback: generic divs with 'job' in class name
                cards = soup.find_all("div", class_=lambda c: c and "job" in c.lower())

            for card in cards:
                title_el = card.select_one("h2.title, h2.title.is-5, h2")
                desc_el = card.select_one("div.content, p, .description, .job-description")

                title = (title_el.get_text(strip=True) if title_el else "Untitled Role")
                description = (desc_el.get_text(strip=True) if desc_el else "").strip()

                # If description is empty, try grabbing paragraphs beneath the card
                if not description:
                    paras = [p.get_text(" ", strip=True) for p in card.find_all("p")]
                    description = " ".join(paras)

                items.append({"title": title, "description": description})

            time.sleep(delay)
        except requests.RequestException as e:
            print(f"[ERROR] {e}")

    return items

def main():
    ap = argparse.ArgumentParser(description="Simple JD scraper (BeautifulSoup).")
    ap.add_argument("--url", required=True, help="Start URL (e.g., Real Python fake jobs).")
    ap.add_argument("--pages", type=int, default=1, help="Number of pages (if site supports pagination).")
    ap.add_argument("--out", default="data/job_descriptions.json", help="Output JSON path.")
    args = ap.parse_args()

    items = scrape_job_descriptions(args.url, args.pages)
    if not items:
        print("No items scraped.")
        return

    out_path = args.out
    # Make sure parent folder exists
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(items)} job descriptions to {out_path}")

if __name__ == "__main__":
    main()

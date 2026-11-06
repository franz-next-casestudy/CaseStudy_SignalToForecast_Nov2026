"""Cross-check the latest Delta earnings release and stored SEC passenger revenue."""

import calendar
import hashlib
import io
import json
import re
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from pypdf import PdfReader

from forecasting import REVENUE, load_financials

LISTING_URL = "https://news.delta.com/category/financial"
MONTHS = {"march": 3, "june": 6, "september": 9, "december": 12}


class ReleaseLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.publication_date = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.links.append(urljoin("https://news.delta.com/", values["href"]))
        candidate = values.get("datetime") if tag == "time" else values.get("content") if tag == "meta" and values.get("property") == "article:published_time" else None
        if candidate and re.match(r"\d{4}-\d{2}-\d{2}", candidate):
            self.publication_date = candidate[:10]


def latest_release(links, as_of):
    candidates = []
    for url in links:
        match = re.fullmatch(r"https://news\.delta\.com/delta-air-lines-announces-(march|june|september|december)-quarter-(\d{4})-financial-results/?", url)
        if match:
            month, year = MONTHS[match[1]], int(match[2])
            end = date(year, month, calendar.monthrange(year, month)[1])
            if end <= as_of:
                candidates.append((pd.Period(end, freq="Q-DEC"), url))
    if not candidates:
        raise ValueError("The official financial listing did not expose a completed-quarter earnings release.")
    return max(candidates, key=lambda item: item[0])


def verify(base_dir, as_of=None, timeout=25):
    base_dir = Path(base_dir)
    as_of = as_of or date.today()
    result = {"status": "unverified", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "as_of_date": as_of.isoformat(), "listing_url": LISTING_URL,
              "note": "Quarter verification has not completed."}
    evidence_dir = base_dir / "output_data" / "verification_sources"

    def fetch(url, suffix):
        response = requests.get(url, timeout=(10, timeout))
        response.raise_for_status()
        if urlparse(response.url).hostname != "news.delta.com":
            raise ValueError("The official source redirected outside Delta's newsroom.")
        digest = hashlib.sha256(response.content).hexdigest()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / f"{digest}.{suffix}").write_bytes(response.content)
        return response.content, digest

    try:
        body, listing_hash = fetch(LISTING_URL, "html")
        listing = ReleaseLinks()
        listing.feed(body.decode("utf-8", errors="replace"))
        quarter, release_url = latest_release(listing.links, as_of)
        body, release_hash = fetch(release_url, "html")
        release = ReleaseLinks()
        release.feed(body.decode("utf-8", errors="replace"))
        pdf_urls = [url for url in release.links if url.startswith("https://news.delta.com/sites/default/files/") and url.lower().endswith(".pdf")]
        if not pdf_urls or not release.publication_date or date.fromisoformat(release.publication_date) > as_of:
            raise ValueError("The release publication date or official financial-tables PDF could not be verified.")
        pdf_url = next((url for url in pdf_urls if "results" in url.lower()), pdf_urls[0])
        pdf, pdf_hash = fetch(pdf_url, "pdf")
        if not pdf.startswith(b"%PDF"):
            raise ValueError("The official financial-tables response is not a PDF.")
        text = " ".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(pdf)).pages)
        values = re.findall(r"\bPassenger(?:\s+revenue)?\s+\$\s*([\d,]+)", text, flags=re.IGNORECASE)
        if not values:
            raise ValueError("A passenger-revenue row could not be identified in the release tables.")
        passenger_revenue = float(values[0].replace(",", ""))
        financials = load_financials(base_dir / "input_data" / "delta_quarterly.csv")
        stored = financials.iloc[-1]
        result.update(latest_reported_quarter=str(quarter), release_url=release_url, release_publication_date=release.publication_date,
                      financial_tables_url=pdf_url, release_passenger_revenue_usd_millions=passenger_revenue,
                      sec_quarter=stored["quarter"], sec_url=stored["source_url"], sec_filing_date=stored["published_date"].isoformat(),
                      sec_passenger_revenue_usd_millions=float(stored[REVENUE]),
                      source_sha256={"listing": listing_hash, "release": release_hash, "financial_tables": pdf_hash,
                                     "sec_document": stored["source_sha256"]},
                      forecast_quarters=[str(quarter + horizon) for horizon in range(1, 7)])
        if stored["quarter"] != str(quarter) or abs(float(stored[REVENUE]) - passenger_revenue) > 0.001:
            result.update(status="mismatch", note="Latest official earnings-release quarter/value and stored SEC history disagree. Update financials before forecasting.")
        else:
            result.update(status="verified", note="Latest completed-quarter release discovered on Delta's official financial listing; passenger revenue agrees with the stored SEC filing. Recheck before submission if new results are released.")
    except (requests.RequestException, ValueError, OSError, KeyError) as error:
        result["note"] = "Official verification could not complete; quarter status remains unverified. " + type(error).__name__
    destination = base_dir / "output_data" / "quarter_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = verify(Path(__file__).parent)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "verified" else 1)
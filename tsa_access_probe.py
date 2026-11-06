"""One-request TSA access test, separate from the forecasting app.

Install: python -m pip install requests pandas lxml
Run:     python tsa_access_probe.py
Optional client: python -m pip install curl_cffi
                python tsa_access_probe.py --client curl
Historical page: python tsa_access_probe.py --client curl --year 2025

Each invocation makes one GET (plus any ordinary redirects), without retries,
and saves the response, diagnostic report, and parsed observations if valid.
No app files, API keys, proxies, or browser cookies are used.
"""

import argparse
import hashlib
import io
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


def parse_counts(html, year):
    """Select the requested year's count column, never comparison-year counts."""
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    observed_headers = []
    for table in tables:
        def label(column):
            parts = column if isinstance(column, tuple) else (column,)
            return " ".join(str(p).strip() for p in parts if not str(p).startswith("Unnamed"))

        labels = [label(c) for c in table.columns]
        observed_headers.append(labels)
        date_indexes = [i for i, name in enumerate(labels) if name.lower() == "date"]
        if len(date_indexes) != 1:
            continue
        year_indexes = [i for i, name in enumerate(labels)
                        if re.search(rf"\b{year}\b", name)]
        if len(year_indexes) == 1:
            count_index = year_indexes[0]
        elif not year_indexes:
            generic = [i for i, name in enumerate(labels)
                       if name.lower() in ("numbers", "number", "traveler throughput")]
            if len(generic) != 1:
                continue
            count_index = generic[0]
        else:
            continue
        dates = pd.to_datetime(table.iloc[:, date_indexes[0]].astype(str).str.strip(),
                               format="%m/%d/%Y", errors="coerce")
        values = table.iloc[:, count_index].astype(str).str.replace(",", "", regex=False).str.strip()
        selected = dates.notna() & dates.dt.year.eq(year)
        if not selected.any():
            continue
        if not values[selected].str.fullmatch(r"[0-9]+(?:\.0+)?").all():
            raise ValueError("A selected count is missing or not a nonnegative integer.")
        result = pd.DataFrame({"date": dates[selected].dt.strftime("%Y-%m-%d"),
                               "passengers": values[selected].astype(float).astype("int64")})
        if result["date"].duplicated().any() or result["passengers"].le(0).any():
            raise ValueError("Duplicate dates or nonpositive counts found.")
        if result["date"].gt(date.today().isoformat()).any():
            raise ValueError("The table contains future observations.")
        return result.sort_values("date").reset_index(drop=True)
    raise ValueError(f"No unambiguous {year} TSA count table found. Headers: {observed_headers}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client", choices=("requests", "curl"), default="requests")
    parser.add_argument("--year", type=int, default=date.today().year)
    parser.add_argument("--output-dir", type=Path, default=Path("tsa_probe_output"))
    args = parser.parse_args()
    if not 2019 <= args.year <= date.today().year:
        parser.error("Choose a year from 2019 through the current year.")
    url = "https://www.tsa.gov/travel/passenger-volumes"
    if args.year != date.today().year:
        url += f"/{args.year}"
    started = datetime.now(timezone.utc)
    output = args.output_dir / f"{args.client}_{args.year}_{started.strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True)
    report = {"client": args.client, "requested_url": url,
              "retrieved_at_utc": started.isoformat(), "success": False}
    response = None
    try:
        if args.client == "curl":
            from curl_cffi import requests as client
            response = client.get(url, impersonate="chrome", timeout=25,
                                  allow_redirects=True, max_redirects=3)
        else:
            import requests as client
            with client.Session() as session:
                session.max_redirects = 3
                response = session.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }, timeout=(10, 25))
        body = response.content
        (output / "response.html").write_bytes(body)
        report.update(http_status=response.status_code, final_url=str(response.url),
                      bytes_received=len(body), source_sha256=hashlib.sha256(body).hexdigest(),
                      response_headers={k: response.headers.get(k) for k in
                                        ("content-type", "server", "date", "retry-after")})
        if response.status_code != 200:
            report["error"] = f"HTTP {response.status_code}; no observations accepted."
        else:
            frame = parse_counts(response.text, args.year)
            latest = frame.iloc[-1]["date"]
            expected_end = f"{args.year}-12-31" if args.year < date.today().year else latest
            expected = set(pd.date_range(f"{args.year}-01-01", expected_end).strftime("%Y-%m-%d"))
            missing = sorted(expected - set(frame["date"]))
            report.update(success=True, rows=len(frame), first_observation=frame.iloc[0]["date"],
                          latest_observation=latest, age_days=(date.today() - date.fromisoformat(latest)).days,
                          missing_dates=len(missing), missing_examples=missing[:5],
                          latest_passengers=int(frame.iloc[-1]["passengers"]))
            frame["source_url"] = str(response.url)
            frame["retrieved_at_utc"] = started.isoformat()
            frame["source_sha256"] = report["source_sha256"]
            frame.to_csv(output / "observations.csv", index=False)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if response is not None:
            response.close()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nDiagnostics saved in: {output}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

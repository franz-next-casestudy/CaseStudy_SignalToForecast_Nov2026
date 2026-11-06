"""Download cutoff-specific ALFRED airfare CPI snapshots without an API key.

Install: python -m pip install requests
One origin: python download_airfare_vintages.py --cutoffs 2024-09-01
All four:   python download_airfare_vintages.py

The cutoff means the START of that date in America/New_York. To avoid
same-day release-time ambiguity, request the previous calendar day's vintage.
The entire available monthly history from 2019 is downloaded for each cutoff;
values are not taken from today's FRED CSV. Nothing is interpolated.

This prepares the AIRFARE input only. It does not change the live dataset,
run forecasts, prove TSA availability, or make the overall backtest compliant.
"""

import argparse
import csv
import hashlib
import io
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests


SERIES = "CUSR0000SETG01"
ENDPOINT = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
DEFAULT_CUTOFFS = ("2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01")
START = date(2019, 1, 1)
CONVENTION = "Start of cutoff date in America/New_York; previous-calendar-day ALFRED vintage; same-day releases excluded"


def next_month(day):
    return date(day.year + (day.month == 12), 1 if day.month == 12 else day.month + 1, 1)


def parse_vintage(body, as_of):
    """Require the exact dated series, not an undated/current-vintage fallback."""
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    date_column = next((name for name in ("observation_date", "DATE") if name in fields), None)
    expected_column = f"{SERIES}_{as_of.strftime('%Y%m%d')}"
    if date_column is None or len(fields) != 2 or set(fields) != {date_column, expected_column}:
        raise ValueError(f"Expected {expected_column}; received columns {fields}. Refusing an unverified vintage.")
    rows = []
    seen = set()
    for item in reader:
        if None in item or any(value is None for value in item.values()):
            raise ValueError("Malformed source CSV row.")
        observed = date.fromisoformat(item[date_column])
        if observed.day != 1 or observed < START or observed > as_of or observed in seen:
            raise ValueError(f"Invalid, duplicate, or out-of-range observation: {observed}")
        seen.add(observed)
        raw = item[expected_column].strip()
        missing = raw in ("", ".")
        if not missing:
            try:
                number = Decimal(raw)
            except InvalidOperation as error:
                raise ValueError(f"Invalid CPI value for {observed}") from error
            if not number.is_finite() or number <= 0:
                raise ValueError(f"Nonpositive or nonfinite CPI value for {observed}")
        rows.append({"date": observed.isoformat(), "airfare_cpi_index": "" if missing else raw,
                     "value_status": "missing_at_source" if missing else "reported"})
    rows.sort(key=lambda row: row["date"])
    if not rows or not any(row["value_status"] == "reported" for row in rows):
        raise ValueError("No valid airfare observations returned.")
    expected = START
    for row in rows:
        if row["date"] != expected.isoformat():
            raise ValueError(f"Missing monthly row {expected}; source blanks must be explicit, not filled.")
        expected = next_month(expected)
    return rows, expected_column


def download_snapshot(session, cutoff, output_root):
    as_of = cutoff - timedelta(days=1)
    retrieved = datetime.now(timezone.utc).isoformat()
    response = session.get(ENDPOINT, params={
        "id": SERIES, "vintage_date": as_of.isoformat(),
        "cosd": START.isoformat(), "coed": as_of.isoformat(),
    }, timeout=(10, 30))
    try:
        response.raise_for_status()
        body = response.content
        rows, column = parse_vintage(body, as_of)
        latest_reported = max(row["date"] for row in rows if row["value_status"] == "reported")
        if (cutoff - date.fromisoformat(latest_reported)).days > 95:
            raise ValueError("Latest nonmissing observation is over 95 days old at this cutoff; inspect the response.")
        source_hash = hashlib.sha256(body).hexdigest()
        metadata = {
            "forecast_cutoff": cutoff.isoformat(), "vintage_as_of_date": as_of.isoformat(),
            "published_date": "",  # Vintage date is NOT the first publication date of every row.
            "availability_status": "present_in_alfred_as_of_snapshot; individual_release_dates_not_in_csv",
            "source_url": f"https://alfred.stlouisfed.org/series?seid={SERIES}",
            "download_url": response.url, "retrieved_at_utc": retrieved,
            "source_sha256": source_hash, "series_id": SERIES,
            "units": "Index 1982-1984=100", "seasonal_adjustment": "Seasonally Adjusted",
            "frequency": "Monthly", "data_source": "US Bureau of Labor Statistics via ALFRED",
        }
        normalized = [{**row, **metadata} for row in rows]
        target = output_root / cutoff.isoformat()
        target.mkdir(parents=True, exist_ok=True)
        raw_name = f"airfare_raw_{source_hash[:16]}.csv"
        (target / raw_name).write_bytes(body)
        normalized_path = target / "airfare_cpi_monthly.csv"
        with normalized_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
            writer.writeheader()
            writer.writerows(normalized)
        summary = {
            "status": "downloaded_and_validated", "series_id": SERIES,
            "forecast_cutoff": cutoff.isoformat(), "cutoff_convention": CONVENTION,
            "vintage_as_of_date": as_of.isoformat(), "returned_series_column": column,
            "http_status": response.status_code, "download_url": response.url,
            "retrieved_at_utc": retrieved, "source_sha256": source_hash,
            "raw_file": raw_name, "normalized_file": normalized_path.name,
            "normalized_sha256": hashlib.sha256(normalized_path.read_bytes()).hexdigest(),
            "monthly_rows": len(rows), "first_observation": rows[0]["date"],
            "latest_observation": rows[-1]["date"], "latest_nonmissing_observation": latest_reported,
            "missing_value_months": [row["date"] for row in rows if row["value_status"] == "missing_at_source"],
            "note": "Airfare information set only. Values reflect this vintage, including revisions already known then. TSA and financial availability must be checked separately.",
        }
        (target / "airfare_snapshot.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        response.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoffs", nargs="+", default=list(DEFAULT_CUTOFFS), metavar="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("input_data/backtest_vintages"))
    args = parser.parse_args()
    try:
        cutoffs = [date.fromisoformat(value) for value in args.cutoffs]
    except ValueError:
        parser.error("Cutoffs must use YYYY-MM-DD.")
    if len(set(cutoffs)) != len(cutoffs) or any(day <= START or day > date.today() for day in cutoffs):
        parser.error("Use distinct cutoff dates after 2019-01-01 and no later than today.")
    print(CONVENTION, flush=True)
    with requests.Session() as session:
        for cutoff in cutoffs:
            try:
                result = download_snapshot(session, cutoff, args.output_dir)
            except (requests.RequestException, ValueError, OSError, InvalidOperation) as error:
                print(f"FAILED {cutoff}: {type(error).__name__}: {error}", flush=True)
                print("Stopped without substituting current-vintage data. Earlier completed snapshots remain available.")
                return 1
            print(json.dumps({key: result[key] for key in (
                "forecast_cutoff", "vintage_as_of_date", "returned_series_column", "monthly_rows",
                "latest_nonmissing_observation", "missing_value_months", "source_sha256")}, indent=2), flush=True)
            print(f"Saved: {args.output_dir / cutoff.isoformat()}", flush=True)
    print("Airfare snapshots ready. Existing live data and forecast outputs were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

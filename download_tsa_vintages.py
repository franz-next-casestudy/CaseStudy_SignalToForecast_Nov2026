"""Build TSA information sets from pre-cutoff Wayback captures, not today's data.

Keep tsa_access_probe.py beside this file (its validated table parser is reused).
Dependencies: requests pandas lxml (already used by tsa_access_probe.py).
First origin: python download_tsa_vintages.py
Four origins: python download_tsa_vintages.py --cutoffs 2024-09-01 2024-12-01 2025-03-01 2025-06-01

Requests are sequential with a default 30-second pause AFTER each response.
Successful source/index downloads are reused. A 429 stops the run and records
a cooldown (at least 15 minutes, or longer if Retry-After requires it). Restarting
the script during that cooldown will not send more requests. Run one copy only.
These are local runtime files, not datasets to commit to Git.

Training history starts in 2021: the model trains from 2022 and needs the prior
year for YoY features. Cutoffs are start-of-day US Eastern. Archive selection
uses the stricter preceding UTC-day boundary, safely before that local cutoff.
Each year must be complete to year-end or its current-year last observation.
Nothing is filled from today's TSA CSV. Missing archives block completion.
"""

import argparse
import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from tsa_access_probe import parse_counts


BASE = "https://www.tsa.gov/travel/passenger-volumes"
CDX = "https://web.archive.org/cdx/search/cdx"


class ArchiveClient:
    def __init__(self, timeout, state_path=None, pause_seconds=30):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SignalToForecastCaseStudy/1.0 (sequential historical-data research)"})
        self.session.max_redirects = 3
        self.timeout = timeout
        self.state_path = Path(state_path) if state_path is not None else None
        self.pause_seconds = pause_seconds
        self.state = {}
        if self.state_path is not None and self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(self.state, dict):
                raise ValueError("Invalid local request-state file; no archive request sent.")
            for key in ("next_request_after", "blocked_until", "consecutive_429"):
                value = self.state.get(key, 0)
                if not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                    raise ValueError("Invalid local request-state values; no archive request sent.")
        self.complete_years = {}

    def save_state(self):
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.state_path)

    @staticmethod
    def retry_after_seconds(value, now):
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, parsed.timestamp() - now)
            except (TypeError, ValueError, OverflowError):
                return 0.0

    def get(self, url, **kwargs):
        now = time.time()
        if self.state.get("blocked_until", 0) > now:
            until = datetime.fromtimestamp(self.state["blocked_until"], timezone.utc).isoformat()
            raise ValueError(f"Archive cooldown until {until} (UTC). No request sent; rerun after that time.")
        remaining = self.state.get("next_request_after", 0) - now
        if remaining > 0:
            print(f"  Waiting {remaining:.0f} seconds before the next archive request...", flush=True)
            time.sleep(remaining)
        try:
            response = self.session.get(url, timeout=(10, self.timeout), **kwargs)
        except requests.RequestException:
            self.state["next_request_after"] = time.time() + self.pause_seconds
            self.save_state()
            raise
        now = time.time()
        self.state["next_request_after"] = now + self.pause_seconds
        if response.status_code == 429:
            count = int(self.state.get("consecutive_429", 0)) + 1
            # Conservative local policy, not a claim about the server's reset time.
            backoff = 900 * 2 ** min(count - 1, 3)
            delay = max(backoff, self.retry_after_seconds(response.headers.get("Retry-After"), now))
            self.state.update(blocked_until=now + delay, consecutive_429=count,
                              last_429_url=url, last_429_at_utc=datetime.fromtimestamp(now, timezone.utc).isoformat())
            self.save_state()
            until = datetime.fromtimestamp(now + delay, timezone.utc).isoformat()
            response.close()
            raise ValueError(f"Archive rate limit (429). Stopped; cooldown until {until} (UTC). No automatic retry. Completed files retained.")
        self.state.update(blocked_until=0, consecutive_429=0)
        self.save_state()
        try:
            response.raise_for_status()
        except requests.RequestException:
            response.close()
            raise
        return response

    def close(self):
        self.session.close()


def capture_time(value):
    if not re.fullmatch(r"\d{14}", str(value)):
        raise ValueError("Invalid archive capture timestamp.")
    return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def validate_year(frame, year, cutoff, captured):
    bound = datetime.combine(cutoff, datetime.min.time(), tzinfo=timezone.utc)
    if captured >= bound:
        raise ValueError("Archive capture is not before the conservative cutoff boundary.")
    dates = pd.to_datetime(frame["date"])
    if not dates.dt.year.eq(year).all() or dates.max().date() > captured.date():
        raise ValueError("Observations fall outside their year or after the archive capture date.")
    last = date(year, 12, 31) if year < cutoff.year else dates.max().date()
    expected = set(pd.date_range(date(year, 1, 1), last).strftime("%Y-%m-%d"))
    if set(frame["date"]) != expected:
        raise ValueError(f"Archive does not contain complete {year} coverage through {last}.")


def year_source(client, year, cutoff, target):
    """Reuse only source bytes already captured before this origin."""
    if year in client.complete_years:
        body, info = client.complete_years[year]
        frame = parse_counts(body.decode("utf-8", errors="replace"), year)
        validate_year(frame, year, cutoff, capture_time(info["capture_timestamp"]))
        return frame, body, dict(info)

    def read_saved(metadata_file):
        info = json.loads(metadata_file.read_text(encoding="utf-8"))
        if info.get("year") != year or Path(info["raw_file"]).name != info["raw_file"]:
            raise ValueError("Unexpected year or path in saved source metadata.")
        raw_path = metadata_file.parent / info["raw_file"]
        body = raw_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != info["source_sha256"]:
            raise ValueError(f"Saved archive bytes failed the hash check for {year}.")
        frame = parse_counts(body.decode("utf-8", errors="replace"), year)
        validate_year(frame, year, cutoff, capture_time(info["capture_timestamp"]))
        return frame, body, info

    metadata_file = target / f"tsa_{year}_source.json"
    if metadata_file.exists():
        frame, body, info = read_saved(metadata_file)
        if year < cutoff.year:
            client.complete_years[year] = (body, info)
        return frame, body, info

    # Reuse complete historical years from earlier origins even after restarting.
    # Never reuse an incomplete current-year page or a post-cutoff capture.
    if year < cutoff.year:
        for previous in sorted(target.parent.glob(f"*/tsa_{year}_source.json"), reverse=True):
            try:
                frame, body, info = read_saved(previous)
            except (ValueError, OSError, KeyError, TypeError):
                continue
            client.complete_years[year] = (body, info)
            print(f"  Reusing validated pre-cutoff {year} source from {previous.parent.name}", flush=True)
            return frame, body, info

    source_url = BASE if year == cutoff.year else f"{BASE}/{year}"
    end = (cutoff - timedelta(days=1)).strftime("%Y%m%d") + "235959"
    index_path = target / f"tsa_{year}_archive_index.json"
    if index_path.exists():
        print(f"  Reusing saved pre-cutoff {year} archive index...", flush=True)
        indexed = json.loads(index_path.read_bytes())
    else:
        print(f"  Finding pre-cutoff {year} capture...", flush=True)
        response = client.get(CDX, params={
            "url": source_url, "output": "json", "filter": ["statuscode:200", "mimetype:text/html"],
            "from": f"{year}0101000000", "to": end, "limit": "-3",
            "fl": "timestamp,original,statuscode,mimetype",
        })
        try:
            index_body = response.content
            indexed = response.json()
        finally:
            response.close()
        index_path.write_bytes(index_body)
    if not isinstance(indexed, list) or len(indexed) < 2 or not isinstance(indexed[0], list):
        raise ValueError(f"No pre-cutoff HTML captures returned for {year}; not substituting live data.")
    candidates = [dict(zip(indexed[0], row)) for row in indexed[1:]]
    if any(row.get("statuscode") != "200" or row.get("mimetype") != "text/html" for row in candidates):
        raise ValueError("Archive index contains unexpected status codes or content types.")
    candidates.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
    last_error = "No usable candidate"
    for candidate in candidates:
        stamp = candidate["timestamp"]
        captured = capture_time(stamp)
        if stamp > end:
            raise ValueError("Archive index returned a capture after the requested bound.")
        original = candidate["original"]
        parsed = urlparse(original)
        if parsed.hostname not in ("www.tsa.gov", "tsa.gov") or parsed.path.rstrip("/") != urlparse(source_url).path.rstrip("/"):
            raise ValueError("Unexpected original source in archive index.")
        replay = f"https://web.archive.org/web/{stamp}id_/{original}"
        print(f"  Reading {year} capture {stamp}...", flush=True)
        response = client.get(replay)
        try:
            match = re.search(r"/web/(\d{14})(?:id_)?/", response.url)
            if urlparse(response.url).hostname != "web.archive.org" or not match or match[1] != stamp:
                raise ValueError("Replay redirected to a different capture; refusing ambiguous availability.")
            body = response.content
        finally:
            response.close()
        try:
            frame = parse_counts(body.decode("utf-8", errors="replace"), year)
            validate_year(frame, year, cutoff, captured)
        except ValueError as error:
            last_error = str(error)
            print(f"  Capture rejected: {last_error}", flush=True)
            continue
        info = {
            "year": year, "source_url": original, "archive_url": replay,
            "capture_timestamp": stamp, "captured_at_utc": captured.isoformat(),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(body).hexdigest(), "rows": len(frame),
            "first_observation": frame.iloc[0]["date"], "latest_observation": frame.iloc[-1]["date"],
        }
        if year < cutoff.year:
            client.complete_years[year] = (body, info)
        return frame, body, info
    raise ValueError(f"No complete usable archive for {year}. Last rejection: {last_error}")


def download_origin(client, cutoff, start_year, output):
    target = output / cutoff.isoformat()
    target.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    for year in range(start_year, cutoff.year + 1):
        frame, body, info = year_source(client, year, cutoff, target)
        raw_name = f"tsa_{year}_{info['capture_timestamp']}_{info['source_sha256'][:16]}.html"
        (target / raw_name).write_bytes(body)
        info["raw_file"] = raw_name
        (target / f"tsa_{year}_source.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        frame = frame.assign(
            published_date="", availability_status="observed_in_pre_cutoff_wayback_capture; first_release_time_not_identified",
            source_url=info["source_url"], archive_url=info["archive_url"],
            captured_at_utc=info["captured_at_utc"], retrieved_at_utc=info["retrieved_at_utc"],
            source_sha256=info["source_sha256"], forecast_cutoff=cutoff.isoformat(),
        )
        frames.append(frame)
        sources.append(info)
        print(f"  {year}: {len(frame)} rows through {info['latest_observation']}", flush=True)
    combined = pd.concat(frames, ignore_index=True).sort_values("date")
    latest = date.fromisoformat(combined.iloc[-1]["date"])
    age = (cutoff - latest).days
    if age > 7:
        raise ValueError(f"Latest archived TSA observation is {age} days old at the origin; the current backtest model requires <=7. No snapshot promoted.")
    path = target / "tsa_daily.csv"
    combined.to_csv(path, index=False)
    summary = {
        "status": "downloaded_and_validated", "forecast_cutoff": cutoff.isoformat(),
        "archive_boundary": "Strictly before cutoff date at 00:00 UTC; conservative relative to 00:00 America/New_York",
        "training_history_start": f"{start_year}-01-01", "rows": len(combined),
        "latest_observation": latest.isoformat(), "age_days_at_cutoff": age,
        "normalized_file": path.name, "normalized_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": sources,
        "note": "Each value comes from a documented pre-cutoff capture, not a current-history download. Complete historical-year captures may be reused at later origins. This is a conservative archived information set, not proof of the exact first publication time or every revision known by the cutoff. Financial and airfare inputs require separate verification.",
    }
    (target / "tsa_snapshot.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoffs", nargs="+", default=["2024-09-01"])
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--output-dir", type=Path, default=Path("input_data/backtest_vintages"))
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--pause-seconds", type=int, default=30,
                        help="Pause after each response (minimum 15 seconds; default 30).")
    args = parser.parse_args()
    try:
        cutoffs = sorted(date.fromisoformat(value) for value in args.cutoffs)
    except ValueError:
        parser.error("Use YYYY-MM-DD cutoffs.")
    if not 2019 <= args.start_year <= min(day.year for day in cutoffs) or any(day > date.today() for day in cutoffs) or len(set(cutoffs)) != len(cutoffs) or not 1 <= args.timeout <= 60 or not 15 <= args.pause_seconds <= 300:
        parser.error("Use distinct past cutoffs, a valid start year from 2019 onward, and a timeout of 1-60 seconds.")
    client = ArchiveClient(args.timeout, args.output_dir / "tsa_archive_request_state.json", args.pause_seconds)
    try:
        for cutoff in cutoffs:
            print(f"TSA historical information set: {cutoff}", flush=True)
            try:
                result = download_origin(client, cutoff, args.start_year, args.output_dir)
            except (requests.RequestException, ValueError, OSError, KeyError) as error:
                print(f"STOPPED: {type(error).__name__}: {error}", flush=True)
                print("Completed source files remain available for resume. No live data was substituted; existing live inputs were not changed.")
                return 1
            print(json.dumps({key: result[key] for key in ("status", "forecast_cutoff", "rows", "latest_observation", "age_days_at_cutoff")}, indent=2), flush=True)
            print(f"Saved: {args.output_dir / cutoff.isoformat()}", flush=True)
    finally:
        client.close()
    print("TSA archive inputs ready for the requested origins. Forecast integration is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

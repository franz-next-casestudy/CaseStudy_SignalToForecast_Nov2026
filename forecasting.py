"""Live TSA/airfare forecasts and origin-specific historical-snapshot backtests."""

import argparse
import hashlib
import io
import json
import math
import re
import tempfile
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from statistics import NormalDist
from urllib.parse import parse_qs, urlparse

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

REVENUE = "passenger_revenue_usd_millions"
MODEL = "seasonal_naive"
DRIVER_MODEL = "tsa_airfare_ridge"
CUTOFF_CONVENTION = "Start of cutoff date, US Eastern; only earlier filing dates admitted"
SNAPSHOT_BASIS = "tsa_pre_cutoff_captures_and_alfred_as_of_vintages"
REQUIRED_COLUMNS = {"quarter", "period_end", "published_date", REVENUE,
                    "source_url", "additional_source_url", "source_sha256"}


def load_financials(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty or not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError("Financial CSV is empty or lacks required revenue/provenance columns.")
    if not frame["quarter"].str.fullmatch(r"\d{4}Q[1-4]").all():
        raise ValueError("Invalid quarter labels.")
    frame["period"] = pd.PeriodIndex(frame["quarter"], freq="Q-DEC")
    if frame["period"].duplicated().any():
        raise ValueError("Duplicate financial quarters.")
    frame = frame.sort_values("period").reset_index(drop=True)
    expected = pd.period_range(frame["period"].iloc[0], frame["period"].iloc[-1], freq="Q-DEC")
    if list(frame["period"]) != list(expected):
        raise ValueError("Financial quarter history has gaps.")
    for column in ("period_end", "published_date"):
        if not frame[column].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError(f"Invalid {column} dates.")
        frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise").dt.date
    ends = frame["period"].map(lambda period: period.end_time.date())
    if not frame["period_end"].eq(ends).all():
        raise ValueError("Quarter labels and period-end dates disagree.")
    if not (frame["published_date"] > frame["period_end"]).all():
        raise ValueError("A financial filing date does not follow its reporting period.")
    frame[REVENUE] = pd.to_numeric(frame[REVENUE], errors="raise").astype(float)
    if not frame[REVENUE].map(lambda value: math.isfinite(value) and value > 0).all():
        raise ValueError("Revenue values must be finite and positive.")
    if not frame["source_url"].str.startswith("https://www.sec.gov/Archives/edgar/data/").all():
        raise ValueError("Financial rows require direct SEC source links.")
    if not frame["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Financial rows require SHA-256 source hashes.")
    additional = frame["additional_source_url"]
    if not (additional.eq("") | additional.str.startswith("https://www.sec.gov/Archives/edgar/data/")).all():
        raise ValueError("Invalid additional SEC source links.")
    return frame


def available_financials(financials, cutoff):
    cutoff_date = date.fromisoformat(str(cutoff))
    available = financials.loc[financials["published_date"] < cutoff_date].sort_values("period")
    if len(available) < 4:
        raise ValueError(f"Fewer than four available quarters before {cutoff_date}.")
    expected = pd.period_range(available["period"].iloc[0], available["period"].iloc[-1], freq="Q-DEC")
    if list(available["period"]) != list(expected):
        raise ValueError(f"Available financial history has gaps before {cutoff_date}.")
    return available


def seasonal_naive(financials, cutoff):
    available = available_financials(financials, cutoff)
    known = {row["period"]: row for row in available.to_dict("records")}
    latest = available["period"].iloc[-1]
    predictions = {}
    for horizon in range(1, 7):
        target = latest + horizon
        reference = target - 4
        if reference in known:
            observed = known[reference]
            value = observed[REVENUE]
            input_quarter = observed["quarter"]
            published = observed["published_date"].isoformat()
            source_url = observed["source_url"]
            additional_url = observed["additional_source_url"]
            source_hash = observed["source_sha256"]
            reference_type = "observed"
        else:
            earlier = predictions[reference]
            value = earlier["forecast_usd_millions"]
            input_quarter = earlier["input_quarter"]
            published = earlier["input_published_date"]
            source_url = earlier["input_source_url"]
            additional_url = earlier["input_additional_source_url"]
            source_hash = earlier["input_source_sha256"]
            reference_type = "recursive_forecast"
        predictions[target] = {
            "cutoff_date": str(cutoff), "model": MODEL, "latest_available_quarter": str(latest),
            "horizon": horizon, "quarter": str(target), "forecast_usd_millions": value,
            "seasonal_reference_quarter": str(reference), "reference_type": reference_type,
            "input_quarter": input_quarter, "input_published_date": published,
            "input_source_url": source_url, "input_additional_source_url": additional_url,
            "input_source_sha256": source_hash,
        }
    return pd.DataFrame(predictions.values())


def default_origins(financials, evaluation_cutoff, count=4):
    if count < 4:
        raise ValueError("Use at least four backtest origins.")
    evaluation_date = date.fromisoformat(str(evaluation_cutoff))
    actuals = available_financials(financials, evaluation_date)
    scored_quarters = set(actuals["quarter"])
    candidates = []
    for year in range(financials["period"].iloc[0].year, evaluation_date.year + 1):
        for month in (3, 6, 9, 12):
            origin = date(year, month, 1)
            if origin >= evaluation_date:
                continue
            try:
                prediction = seasonal_naive(financials, origin)
            except ValueError:
                continue
            expected_latest = pd.Period(origin, freq="Q-DEC") - 1
            if prediction["latest_available_quarter"].iloc[0] != str(expected_latest):
                continue
            if set(prediction["quarter"]).issubset(scored_quarters):
                candidates.append(origin.isoformat())
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} fully scored calendar origins are available; need {count}.")
    return candidates[-count:]


def backtest(financials, origins, evaluation_cutoff):
    evaluation_date = date.fromisoformat(str(evaluation_cutoff))
    if not origins or len(set(origins)) != len(origins):
        raise ValueError("Backtest origins must be nonempty and unique.")
    actuals = available_financials(financials, evaluation_date).set_index("quarter")
    records = []
    for origin in sorted(origins):
        if date.fromisoformat(str(origin)) >= evaluation_date:
            raise ValueError("Backtest origins must precede the evaluation cutoff.")
        prediction = seasonal_naive(financials, origin)
        if not set(prediction["quarter"]).issubset(actuals.index):
            raise ValueError(f"Not all six outcomes are available for origin {origin}.")
        for row in prediction.to_dict("records"):
            actual = actuals.loc[row["quarter"]]
            value = actual[REVENUE]
            error = row["forecast_usd_millions"] - value
            records.append({**row, "actual_usd_millions": value,
                            "actual_published_date": actual["published_date"].isoformat(),
                            "actual_source_url": actual["source_url"],
                            "actual_source_sha256": actual["source_sha256"],
                            "error_usd_millions": error, "absolute_error_usd_millions": abs(error),
                            "percentage_error": 100 * error / value,
                            "absolute_percentage_error": 100 * abs(error) / value})
    detail = pd.DataFrame(records)
    metrics = detail.groupby("horizon", as_index=False).agg(
        origins=("cutoff_date", "nunique"), mape_percent=("absolute_percentage_error", "mean"),
        mae_usd_millions=("absolute_error_usd_millions", "mean"),
        bias_percent=("percentage_error", "mean"))
    metrics.insert(0, "model", MODEL)
    return detail, metrics


def load_signals(tsa_path, airfare_path):
    signals = {}
    for name, path, column in (("tsa", tsa_path, "passengers"), ("airfare", airfare_path, "airfare_cpi_index")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"date", column, "source_url", "source_sha256"}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"Invalid or empty {name} CSV.")
        dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        if dates.duplicated().any():
            raise ValueError(f"Duplicate {name} observation dates.")
        if name == "airfare" and not dates.dt.day.eq(1).all():
            raise ValueError("Airfare observations must label the first day of a month.")
        values = pd.to_numeric(frame[column].replace("", np.nan), errors="raise").astype(float)
        missing = frame[column].eq("")
        allowed = (missing & frame["value_status"].eq("missing_at_source")
                   if name == "airfare" and "value_status" in frame else pd.Series(False, index=frame.index))
        valid = values.map(lambda value: math.isfinite(value) and value > 0)
        if not (valid | allowed).all():
            raise ValueError(f"Invalid {name} observations; source blanks must be explicitly flagged.")
        if not frame["source_url"].str.startswith("https://").all() or not frame["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Missing {name} provenance.")
        frame["date"] = dates
        frame["value"] = values
        signals[name] = frame.sort_values("date").reset_index(drop=True)
    return signals


def _artifact(directory, filename, expected_hash):
    """Require a preserved same-directory artifact with the declared bytes."""
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Snapshot metadata contains an invalid artifact filename.")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("Snapshot metadata lacks a valid SHA-256 hash.")
    path = Path(directory) / filename
    if path.resolve().parent != Path(directory).resolve():
        raise ValueError("Snapshot artifact resolves outside its directory.")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_hash:
        raise ValueError(f"Snapshot artifact hash mismatch: {filename}")
    return body


class _ArchiveRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            if self.row is not None:
                self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _archived_counts(body, year):
    parser = _ArchiveRows()
    parser.feed(body.decode("utf-8", errors="replace"))
    count_index, values = None, {}
    for cells in parser.rows:
        if cells and cells[0].lower() == "date":
            choices = [i for i, name in enumerate(cells) if re.search(rf"\b{year}\b", name)]
            if not choices:
                choices = [i for i, name in enumerate(cells) if name.lower() in ("numbers", "number", "traveler throughput")]
            count_index = choices[0] if len(choices) == 1 else None
            continue
        if count_index is None or len(cells) <= count_index:
            continue
        try:
            observed = datetime.strptime(cells[0], "%m/%d/%Y").date()
        except ValueError:
            continue
        if observed.year != year:
            continue
        raw = cells[count_index].replace(",", "").replace(" ", "")
        if not re.fullmatch(r"\d+", raw) or int(raw) <= 0 or observed.isoformat() in values:
            raise ValueError("Invalid or duplicate passenger count in preserved TSA HTML.")
        values[observed.isoformat()] = int(raw)
    if not values:
        raise ValueError(f"No unambiguous {year} count table in preserved TSA HTML.")
    return values


def load_vintage_signals(vintage_dir, cutoff):
    """Fail closed: never replace absent/invalid historical evidence with live data."""
    cutoff_date = date.fromisoformat(str(cutoff))
    root = Path(vintage_dir) / cutoff_date.isoformat()
    manifests = {}
    for name in ("tsa", "airfare"):
        path = root / f"{name}_snapshot.json"
        manifests[name] = json.loads(path.read_text(encoding="utf-8"))
        manifest = manifests[name]
        if manifest.get("forecast_cutoff") != cutoff_date.isoformat() or manifest.get("status") != "downloaded_and_validated":
            raise ValueError(f"Wrong cutoff or incomplete {name} snapshot in {root.name}.")
        filename = "tsa_daily.csv" if name == "tsa" else "airfare_cpi_monthly.csv"
        if manifest.get("normalized_file") != filename:
            raise ValueError(f"Unexpected normalized {name} snapshot filename.")
        _artifact(root, filename, manifest.get("normalized_sha256"))
    signals = load_signals(root / "tsa_daily.csv", root / "airfare_cpi_monthly.csv")
    for name, frame in signals.items():
        if not frame["date"].lt(pd.Timestamp(cutoff_date)).all():
            raise ValueError(f"Post-cutoff observations in {name} snapshot.")
        if "forecast_cutoff" not in frame or not frame["forecast_cutoff"].eq(cutoff_date.isoformat()).all():
            raise ValueError(f"Per-row cutoff mismatch in {name} snapshot.")

    airfare = manifests["airfare"]
    as_of = cutoff_date - timedelta(days=1)
    series = "CUSR0000SETG01"
    column = f"{series}_{as_of.strftime('%Y%m%d')}"
    if airfare.get("series_id") != series or airfare.get("vintage_as_of_date") != as_of.isoformat() or airfare.get("returned_series_column") != column:
        raise ValueError("Airfare metadata does not identify the requested previous-day vintage.")
    download = urlparse(airfare.get("download_url", ""))
    params = parse_qs(download.query)
    if download.hostname != "alfred.stlouisfed.org" or download.path != "/graph/alfredgraph.csv" or params.get("id") != [series] or params.get("vintage_date") != [as_of.isoformat()]:
        raise ValueError("Airfare vintage source URL does not match the declared series/date.")
    raw = _artifact(root, airfare.get("raw_file"), airfare.get("source_sha256"))
    source = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    date_column = next((name for name in ("observation_date", "DATE") if name in source), None)
    if date_column is None or set(source.columns) != {date_column, column}:
        raise ValueError("Preserved ALFRED response has an unexpected vintage column.")
    source_values = source[column].str.strip().replace(".", "")
    if source[date_column].duplicated().any():
        raise ValueError("Duplicate observation months in preserved ALFRED response.")
    expected = dict(zip(source[date_column], source_values))
    observed = dict(zip(signals["airfare"]["date"].dt.strftime("%Y-%m-%d"), signals["airfare"]["airfare_cpi_index"]))
    if observed != expected or len(observed) != airfare.get("monthly_rows"):
        raise ValueError("Airfare CSV values differ from the preserved vintage response.")
    frame = signals["airfare"]
    if not frame["source_sha256"].eq(airfare["source_sha256"]).all() or "vintage_as_of_date" not in frame or not frame["vintage_as_of_date"].eq(as_of.isoformat()).all():
        raise ValueError("Airfare per-row vintage provenance mismatch.")
    if list(frame["date"]) != list(pd.date_range(frame["date"].iloc[0], frame["date"].iloc[-1], freq="MS")):
        raise ValueError("Airfare snapshot has absent monthly rows; source blanks must be explicit.")

    tsa = manifests["tsa"]
    sources = tsa.get("sources", [])
    if not sources:
        raise ValueError("TSA snapshot has no archived source evidence.")
    frame = signals["tsa"]
    if not frame["value"].eq(np.floor(frame["value"])).all():
        raise ValueError("TSA vintage counts must be integers.")
    source_years, capture_dates = set(), []
    bound = datetime.combine(cutoff_date, datetime.min.time(), tzinfo=timezone.utc)
    for item in sources:
        year = item.get("year")
        if not isinstance(year, int) or year in source_years:
            raise ValueError("Invalid or duplicate TSA source year.")
        source_years.add(year)
        stamp = item.get("capture_timestamp", "")
        if not re.fullmatch(r"\d{14}", stamp):
            raise ValueError("Invalid TSA archive capture timestamp.")
        captured = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        if captured >= bound or item.get("captured_at_utc") != captured.isoformat():
            raise ValueError("TSA capture is not a matching pre-cutoff timestamp.")
        original = item.get("source_url", "")
        original_url = urlparse(original)
        if original_url.hostname not in ("www.tsa.gov", "tsa.gov") or original_url.path.rstrip("/") not in ("/travel/passenger-volumes", f"/travel/passenger-volumes/{year}"):
            raise ValueError("Unexpected original TSA source URL.")
        if item.get("archive_url") != f"https://web.archive.org/web/{stamp}id_/{original}":
            raise ValueError("TSA archive URL does not match its timestamp/source.")
        counts = _archived_counts(_artifact(root, item.get("raw_file"), item.get("source_sha256")), year)
        subset = frame.loc[frame["date"].dt.year == year]
        observed = dict(zip(subset["date"].dt.strftime("%Y-%m-%d"), subset["value"].astype(int)))
        if observed != counts or len(observed) != item.get("rows"):
            raise ValueError(f"TSA {year} CSV values differ from the preserved HTML.")
        if not subset["source_sha256"].eq(item["source_sha256"]).all() or "captured_at_utc" not in subset or not subset["captured_at_utc"].eq(captured.isoformat()).all():
            raise ValueError("TSA per-row archive provenance mismatch.")
        if subset["date"].max().date() > captured.date():
            raise ValueError("TSA observations postdate their capture.")
        expected_end = date(year, 12, 31) if year < cutoff_date.year else subset["date"].max().date()
        if set(subset["date"].dt.strftime("%Y-%m-%d")) != set(pd.date_range(date(year, 1, 1), expected_end).strftime("%Y-%m-%d")):
            raise ValueError("TSA source year has incomplete coverage.")
        capture_dates.append(captured.isoformat())
    if set(frame["date"].dt.year) != source_years or len(frame) != tsa.get("rows"):
        raise ValueError("TSA snapshot coverage does not match its source manifest.")
    if list(frame["date"]) != list(pd.date_range(frame["date"].iloc[0], frame["date"].iloc[-1], freq="D")):
        raise ValueError("TSA vintage contains missing daily rows.")
    if frame["date"].iloc[-1].date().isoformat() != tsa.get("latest_observation"):
        raise ValueError("TSA manifest latest observation does not match its CSV.")
    audit = {
        "basis": SNAPSHOT_BASIS, "signals_verified": True,
        "tsa_latest_observation": tsa["latest_observation"],
        "tsa_age_days_at_cutoff": (cutoff_date - frame["date"].iloc[-1].date()).days,
        "tsa_capture_times_utc": sorted(capture_dates),
        "airfare_vintage_as_of_date": as_of.isoformat(),
        "airfare_latest_observation": signals["airfare"]["date"].iloc[-1].date().isoformat(),
        "snapshot_csv_sha256": {name: manifests[name]["normalized_sha256"] for name in ("tsa", "airfare")},
        "snapshot_manifest_sha256": {name: hashlib.sha256((root / f"{name}_snapshot.json").read_bytes()).hexdigest() for name in ("tsa", "airfare")},
    }
    return signals, audit


def financial_origin_audit(financials, cutoff, input_path=None):
    """Record admitted filings; additionally verify extraction evidence/source bytes when present."""
    known = available_financials(financials, cutoff)
    cutoff_date = date.fromisoformat(str(cutoff))
    audit = {"latest_available_quarter": known.iloc[-1]["quarter"],
             "latest_admitted_filing_date": max(known["published_date"]).isoformat(),
             "filing_dates_before_cutoff": True, "source_documents_verified": False,
             "financial_inputs": [{"quarter": row["quarter"], "filing_date": row["published_date"].isoformat(),
                                   "source_url": row["source_url"], "source_sha256": row["source_sha256"]}
                                  for row in known.to_dict("records")]}
    evidence_path = Path(input_path).parent / "delta_evidence.csv" if input_path is not None else None
    if evidence_path is None or not evidence_path.exists():
        audit["source_check"] = "Extraction evidence not present; financial source-byte verification remains outstanding."
        return audit
    evidence = pd.read_csv(evidence_path, dtype=str, keep_default_na=False)
    required = {"quarter", "result_metric", "result_value", "filed_date", "source_url", "source_sha256", "local_file"}
    if not required.issubset(evidence.columns):
        raise ValueError("Financial extraction evidence lacks required provenance columns.")
    evidence = evidence.loc[evidence["result_metric"] == REVENUE]
    missing, checked = set(), {}
    sec_dir = evidence_path.parent / "sec"
    for row in known.to_dict("records"):
        inputs = evidence.loc[evidence["quarter"] == row["quarter"]]
        if inputs.empty:
            raise ValueError(f"Missing financial extraction evidence for {row['quarter']}.")
        filed = pd.to_datetime(inputs["filed_date"], format="%Y-%m-%d", errors="raise").dt.date
        if not (filed < cutoff_date).all() or filed.max() != row["published_date"]:
            raise ValueError(f"Financial evidence filing dates disagree with the origin for {row['quarter']}.")
        if not pd.to_numeric(inputs["result_value"], errors="raise").map(lambda value: math.isclose(value, row[REVENUE], abs_tol=1e-6, rel_tol=0)).all():
            raise ValueError(f"Financial result differs from extraction evidence for {row['quarter']}.")
        expected_urls = {row["source_url"]} | ({row["additional_source_url"]} if row["additional_source_url"] else set())
        if set(inputs["source_url"]) != expected_urls or row["source_sha256"] not in set(inputs["source_sha256"]):
            raise ValueError(f"Financial source links/hashes disagree for {row['quarter']}.")
        for item in inputs.to_dict("records"):
            local = Path(item["local_file"].replace("\\", "/"))
            source = sec_dir / local
            if local.is_absolute() or not source.resolve().is_relative_to(sec_dir.resolve()):
                raise ValueError("Financial source path resolves outside the SEC directory.")
            if not source.exists():
                missing.add(item["source_url"])
                continue
            if source not in checked:
                checked[source] = hashlib.sha256(source.read_bytes()).hexdigest()
            if checked[source] != item["source_sha256"]:
                raise ValueError("Financial source-document hash differs from extraction evidence.")
    audit.update(source_documents_verified=not missing, source_documents_checked=len(checked),
                 missing_source_urls=sorted(missing),
                 extraction_evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                 source_check="Extraction evidence and preserved source bytes verified." if not missing else "Evidence dates/values checked; some original SEC source bytes are not locally available.")
    return audit


def quarterly_signals(signals, cutoff):
    series = {}
    for name, frame in signals.items():
        eligible = frame.loc[frame["date"] < pd.Timestamp(cutoff)].copy()
        if eligible.empty:
            raise ValueError(f"No {name} observations before {cutoff}.")
        eligible["period"] = eligible["date"].dt.to_period("Q-DEC")
        values = {}
        for period, group in eligible.groupby("period"):
            expected = (period.end_time.date() - period.start_time.date()).days + 1 if name == "tsa" else 3
            complete = period.end_time < pd.Timestamp(cutoff) and len(group) == expected and group["value"].notna().all()
            values[period] = (group["value"].sum() if name == "tsa" else group["value"].mean()) if complete else np.nan
        series[name] = pd.Series(values, dtype=float)
    quarters = pd.concat(series, axis=1).sort_index()
    return quarters.reindex(pd.period_range(quarters.index.min(), quarters.index.max(), freq="Q-DEC"))


def fit_driver(financials, signals, cutoff):
    available = available_financials(financials, cutoff)
    quarters = quarterly_signals(signals, cutoff)
    features = 100 * np.log(quarters / quarters.shift(4))
    revenue = available.set_index("period")[REVENUE]
    training = features.join((100 * np.log(revenue / revenue.shift(4))).rename("revenue_log_growth"))
    training = training.loc[training.index >= pd.Period("2022Q1", freq="Q-DEC")].dropna().tail(12)
    if len(training) < 8:
        raise ValueError(f"Need eight complete post-2021 training quarters before {cutoff}; found {len(training)}.")
    model = Ridge(alpha=1.0, positive=True, max_iter=10000, tol=1e-8)
    model.fit(training[["tsa", "airfare"]], training["revenue_log_growth"])
    residuals = training["revenue_log_growth"].to_numpy() - model.predict(training[["tsa", "airfare"]])
    fit = {"intercept_log_pp": float(model.intercept_), "tsa_coefficient": float(model.coef_[0]),
           "airfare_coefficient": float(model.coef_[1]), "ridge_alpha": 1.0,
           "training_quarters": len(training), "training_start": str(training.index.min()),
           "training_end": str(training.index.max()), "residual_log_sigma": float(np.std(residuals, ddof=3) / 100)}
    return fit, training, quarters


def driver_assumptions(signals, quarters, cutoff, allow_stale_tsa=False):
    cutoff_time = pd.Timestamp(cutoff)
    tsa = signals["tsa"].loc[signals["tsa"]["date"] < cutoff_time].set_index("date")["value"]
    recent = tsa.tail(28)
    if len(recent) != 28 or not recent.index.equals(pd.date_range(recent.index[0], recent.index[-1], freq="D")):
        raise ValueError("The latest 28 TSA days must be complete.")
    tsa_age_days = (cutoff_time - recent.index[-1]).days
    if tsa_age_days > 7 and not allow_stale_tsa:
        raise ValueError("TSA observations are stale relative to the forecast cutoff.")
    prior = tsa.reindex(recent.index - pd.Timedelta(days=364))
    if prior.isna().any():
        raise ValueError("The comparable 28 TSA days one year earlier must be complete.")
    tsa_seed = 100 * math.log(recent.sum() / prior.sum())
    airfare = signals["airfare"].loc[signals["airfare"]["date"] < cutoff_time].set_index("date")["value"]
    pairs = []
    for observed, value in airfare.dropna().items():
        previous = airfare.get(observed - pd.DateOffset(years=1), np.nan)
        if observed + pd.offsets.MonthEnd(0) < cutoff_time and pd.notna(previous):
            pairs.append((observed, value, previous))
    pairs = pairs[-3:]
    if len(pairs) != 3 or (cutoff_time - pairs[-1][0]).days > 95:
        raise ValueError("Need three recent airfare months with valid year-earlier comparators.")
    airfare_seed = 100 * math.log(sum(item[1] for item in pairs) / sum(item[2] for item in pairs))
    growth = 100 * np.log(quarters / quarters.shift(4))
    limits = {"tsa": (0.97, 1.06), "airfare": (0.97, 1.04)}
    assumptions = {}
    for name, seed in (("tsa", tsa_seed), ("airfare", airfare_seed)):
        history = growth[name].dropna().tail(8)
        if len(history) < 4:
            raise ValueError(f"Insufficient quarterly {name} history for a structural assumption.")
        low, high = limits[name]
        raw_terminal = float(history.median())
        terminal = float(np.clip(raw_terminal, 100 * math.log(low), 100 * math.log(high)))
        bounded_seed = float(np.clip(seed, 100 * math.log(0.8), 100 * math.log(1.2)))
        assumptions[name] = {"raw_seed_log_pp": seed, "seed_log_pp": bounded_seed,
                             "raw_terminal_log_pp": raw_terminal, "terminal_log_pp": terminal,
                             "seed_clipped": seed != bounded_seed, "terminal_clipped": raw_terminal != terminal}
    assumptions["tsa"].update(latest_observation=recent.index[-1].date().isoformat(),
                               observation_age_days=tsa_age_days, stale=tsa_age_days > 7,
                               seed_window_start=recent.index[0].date().isoformat(),
                               comparable_window_start=prior.index[0].date().isoformat())
    assumptions["airfare"].update(latest_observation=pairs[-1][0].date().isoformat(),
                                  seed_months=[item[0].date().isoformat() for item in pairs])
    return assumptions


def driver_forecast(financials, signals, cutoff, volume_shift_pp=0.0, price_shift_pp=0.0, signal_actions=None,
                    allow_stale_tsa=False):
    actions = {"tsa": "use_recent", "airfare": "use_recent"} if signal_actions is None else signal_actions
    if not isinstance(actions, dict) or set(actions) != {"tsa", "airfare"} or any(action not in ("use_recent", "use_structural") for action in actions.values()):
        raise ValueError("Signal actions must specify use_recent or use_structural for TSA and airfare.")
    fit, training, quarters = fit_driver(financials, signals, cutoff)
    assumptions = driver_assumptions(signals, quarters, cutoff, allow_stale_tsa)
    if assumptions["tsa"]["stale"] and actions["tsa"] != "use_structural":
        raise ValueError("Older TSA data requires the longer-run volume assumption.")
    baseline = seasonal_naive(financials, cutoff)
    available = available_financials(financials, cutoff)
    known = available.set_index("quarter")[REVENUE].to_dict()
    rows = []
    for row in baseline.to_dict("records"):
        weight = (6 - row["horizon"]) / 5
        features = {}
        for name, shift in (("tsa", volume_shift_pp), ("airfare", price_shift_pp)):
            spec = assumptions[name]
            signal_weight = weight if actions[name] == "use_recent" else 0
            log_growth = signal_weight * spec["seed_log_pp"] + (1 - signal_weight) * spec["terminal_log_pp"]
            arithmetic_growth = math.expm1(log_growth / 100) + shift / 100
            if not math.isfinite(arithmetic_growth) or arithmetic_growth <= -1:
                raise ValueError("Driver growth assumptions must be finite and greater than -100%.")
            features[name] = 100 * math.log1p(arithmetic_growth)
        intercept = weight * fit["intercept_log_pp"]
        volume_contribution = fit["tsa_coefficient"] * features["tsa"]
        price_contribution = fit["airfare_coefficient"] * features["airfare"]
        raw_growth = intercept + volume_contribution + price_contribution
        applied_growth = float(np.clip(raw_growth, 100 * math.log(0.7), 100 * math.log(1.3)))
        reference = row["seasonal_reference_quarter"]
        anchor = known[reference]
        value = anchor * math.exp(applied_growth / 100)
        known[row["quarter"]] = value
        rows.append({**row, "model": DRIVER_MODEL, "forecast_usd_millions": value,
                     "tsa_action": actions["tsa"], "airfare_action": actions["airfare"],
                     "seasonal_anchor_usd_millions": anchor, "near_term_weight": weight,
                     "tsa_growth_percent": 100 * math.expm1(features["tsa"] / 100),
                     "airfare_growth_percent": 100 * math.expm1(features["airfare"] / 100),
                     "intercept_contribution_log_pp": intercept, "tsa_contribution_log_pp": volume_contribution,
                     "airfare_contribution_log_pp": price_contribution,
                     "raw_revenue_growth_percent": 100 * math.expm1(raw_growth / 100),
                     "revenue_growth_percent": 100 * math.expm1(applied_growth / 100),
                     "growth_guardrail_hit": raw_growth != applied_growth,
                     "signal_data_basis": "current_vintage_observation_date_cutoff"})
    return pd.DataFrame(rows), fit, training, assumptions


def add_driver_ranges(forecast, financials, cutoff, fit):
    result = forecast.copy()
    scale = available_financials(financials, cutoff)[REVENUE].tail(4).mean()
    sigma = max(0.05, fit["residual_log_sigma"])
    width = scale * NormalDist().inv_cdf(0.9) * sigma * np.sqrt(result["horizon"])
    result["lower_80_usd_millions"] = np.maximum(0, result["forecast_usd_millions"] - width)
    result["upper_80_usd_millions"] = result["forecast_usd_millions"] + width
    result["interval_half_width_usd_millions"] = width
    result["interval_sigma"] = sigma
    result["interval_method"] = "illustrative_normal_residual_sqrt_h; 5pct_floor; not_coverage_calibrated"
    return result


def backtest_driver(financials, signals, origins, evaluation_cutoff, vintage_dir=None, input_path=None):
    baseline_detail, baseline_metrics = backtest(financials, origins, evaluation_cutoff)
    actual_columns = [column for column in baseline_detail if column.startswith("actual_")]
    frames = []
    fits = []
    for origin in sorted(origins):
        signal_audit = {"basis": "current_vintage_observation_date_cutoff", "signals_verified": False}
        origin_signals = signals
        if vintage_dir is not None:
            try:
                origin_signals, signal_audit = load_vintage_signals(vintage_dir, origin)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Historical signal evidence failed for {origin}: {error}. No current-vintage fallback was used.") from error
        financial_audit = financial_origin_audit(financials, origin, input_path if vintage_dir is not None else None)
        forecast, fit, training, assumptions = driver_forecast(financials, origin_signals, origin)
        forecast = add_driver_ranges(forecast, financials, origin, fit)
        if vintage_dir is not None:
            forecast["signal_data_basis"] = SNAPSHOT_BASIS
            forecast["tsa_observation_age_days"] = signal_audit["tsa_age_days_at_cutoff"]
            forecast["airfare_vintage_as_of_date"] = signal_audit["airfare_vintage_as_of_date"]
        outcomes = baseline_detail.loc[baseline_detail["cutoff_date"] == str(origin), ["quarter", "horizon", *actual_columns]]
        scored = forecast.merge(outcomes, on=["quarter", "horizon"], validate="one_to_one")
        if len(scored) != 6:
            raise ValueError(f"Driver and baseline targets disagree for origin {origin}.")
        scored["error_usd_millions"] = scored["forecast_usd_millions"] - scored["actual_usd_millions"]
        scored["absolute_error_usd_millions"] = scored["error_usd_millions"].abs()
        scored["percentage_error"] = 100 * scored["error_usd_millions"] / scored["actual_usd_millions"]
        scored["absolute_percentage_error"] = scored["percentage_error"].abs()
        scored["inside_80_range"] = scored["actual_usd_millions"].between(scored["lower_80_usd_millions"], scored["upper_80_usd_millions"])
        scored["interval_width_usd_millions"] = scored["upper_80_usd_millions"] - scored["lower_80_usd_millions"]
        frames.append(scored)
        fits.append({"cutoff_date": str(origin), "fit": fit, "assumptions": assumptions,
                     "training_quarters": [str(period) for period in training.index],
                     "signal_audit": signal_audit, "financial_audit": financial_audit})
    detail = pd.concat(frames, ignore_index=True)
    metrics = detail.groupby("horizon", as_index=False).agg(
        origins=("cutoff_date", "nunique"), mape_percent=("absolute_percentage_error", "mean"),
        mae_usd_millions=("absolute_error_usd_millions", "mean"), bias_percent=("percentage_error", "mean"),
        empirical_range_coverage=("inside_80_range", "mean"),
        mean_interval_width_usd_millions=("interval_width_usd_millions", "mean"))
    metrics.insert(0, "model", DRIVER_MODEL)
    return detail, pd.concat([baseline_metrics, metrics], ignore_index=True), fits


def run_driver_model(input_path, tsa_path, airfare_path, output_dir, cutoff, origins=None, origin_count=4,
                     vintage_dir=None):
    if date.fromisoformat(str(cutoff)) > date.today():
        raise ValueError("The cutoff cannot be in the future.")
    financials = load_financials(input_path)
    signals = load_signals(tsa_path, airfare_path)
    automatic_origins = origins is None
    if automatic_origins:
        origins = default_origins(financials, cutoff, origin_count)
    if len(origins) < 4:
        raise ValueError("Use at least four common backtest origins.")
    forecast, fit, training, assumptions = driver_forecast(financials, signals, cutoff)
    forecast = add_driver_ranges(forecast, financials, cutoff, fit)
    detail, comparison, origin_fits = backtest_driver(financials, signals, origins, cutoff, vintage_dir, input_path)
    strict_backtest = vintage_dir is not None and all(item["financial_audit"]["source_documents_verified"] for item in origin_fits)
    sensitivities = []
    for scenario, volume, price in (("base", 0, 0), ("volume_plus_1pp", 1, 0), ("volume_minus_1pp", -1, 0),
                                    ("pricing_plus_1pp", 0, 1), ("pricing_minus_1pp", 0, -1)):
        path, scenario_fit, scenario_training, scenario_assumptions = driver_forecast(financials, signals, cutoff, volume, price)
        sensitivity = path[["horizon", "quarter", "forecast_usd_millions"]].copy()
        sensitivity.insert(0, "scenario", scenario)
        sensitivity["volume_growth_shift_pp"] = volume
        sensitivity["airfare_growth_shift_pp"] = price
        sensitivity["base_forecast_usd_millions"] = forecast["forecast_usd_millions"].to_numpy()
        sensitivity["change_usd_millions"] = sensitivity["forecast_usd_millions"] - sensitivity["base_forecast_usd_millions"]
        sensitivity["change_percent"] = 100 * sensitivity["change_usd_millions"] / sensitivity["base_forecast_usd_millions"]
        sensitivities.append(sensitivity)
    sensitivity_table = pd.concat(sensitivities, ignore_index=True)
    training = training.copy()
    training["fitted_revenue_log_growth"] = fit["intercept_log_pp"] + fit["tsa_coefficient"] * training["tsa"] + fit["airfare_coefficient"] * training["airfare"]
    training["residual_log_pp"] = training["revenue_log_growth"] - training["fitted_revenue_log_growth"]
    training["financial_filing_date"] = financials.set_index("period")["published_date"].reindex(training.index)
    training = training.rename_axis("quarter").reset_index()
    warnings = []
    for name in ("tsa", "airfare"):
        for field in ("seed_clipped", "terminal_clipped"):
            if assumptions[name][field]:
                warnings.append(f"{name}: {field}; inspect raw and applied assumptions.")
    if forecast["growth_guardrail_hit"].any():
        warnings.append("At least one revenue growth projection hit the -30%/+30% limit.")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff_date": str(cutoff),
        "financial_cutoff_convention": CUTOFF_CONVENTION,
        "signal_cutoff_convention": "Origin-specific pre-cutoff TSA captures and previous-day ALFRED vintages; completed quarters only for training" if vintage_dir is not None else "Observation dates strictly before cutoff; current vintage; completed quarters only for training",
        "model": DRIVER_MODEL, "units": "USD millions", "fit": fit, "assumptions": assumptions,
        "input_csv_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for path in (input_path, tsa_path, airfare_path)},
        "headline_quarter": forecast.iloc[-1]["quarter"],
        "headline_forecast_usd_millions": float(forecast.iloc[-1]["forecast_usd_millions"]),
        "origin_selection": "Same automatic calendar origins as baseline, not selected by model error" if automatic_origins else "Explicit common cutoffs",
        "backtest_origins": origin_fits, "external_signals_used": ["tsa", "airfare"],
        "signal_data_basis": SNAPSHOT_BASIS if vintage_dir is not None else "current_vintage_observation_date_cutoff",
        "live_forecast_signal_data_basis": "current_vintage_observation_date_cutoff",
        "historical_signal_evidence_verified": vintage_dir is not None,
        "financial_source_documents_verified": strict_backtest,
        "strict_point_in_time": strict_backtest,
        "strict_point_in_time_scope": "Numerical backtest input evidence; does not backtest LLM decisions or establish earliest earnings-release availability.",
        "submission_ready": False, "approval_status": "not_reviewed", "warnings": warnings,
        "model_contract": {
            "equation": "100*log(revenue/revenue_four_quarters_ago) = intercept + beta_tsa*100*log(TSA/TSA_four_quarters_ago) + beta_airfare*100*log(CPI/CPI_four_quarters_ago)",
            "training": "Last 12 joint complete quarters from 2022Q1 onward; minimum 8; nonnegative Ridge coefficients, alpha=1; not tuned on evaluation errors.",
            "aggregation": "TSA daily passenger sum, airfare three-month mean; incomplete or source-blank quarters excluded, never interpolated.",
            "near_term": "TSA latest 28 complete days versus 364 days earlier (weekday-aligned); airfare last 3 reported monthly pairs with year-earlier comparators. Seed growth bounded to -20%/+20%.",
            "long_term": "Median last 8 available quarterly YoY log growth rates; TSA bounded to -3%/+6%, airfare to -3%/+4%. Seeds and fitted intercept fade linearly over horizons 1-6; intercept is zero at horizon 6.",
            "recursion": "Apply projected YoY revenue growth to the known year-earlier quarter or the earlier forecast at horizons 5-6; annual revenue growth bounded to -30%/+30%.",
            "ranges": "Illustrative nominal 80% normal ranges: last four known quarters' mean revenue * 1.28155 * max(training log-residual sigma, 5%) * sqrt(h). Nonnegative lower bound. No held-out coverage calibration.",
            "sensitivities": "Persistent +/-1 arithmetic percentage point to projected volume or airfare YoY growth at every horizon; model coefficients unchanged.",
        },
        "limitations": [
            ("Historical signal values are checked against pre-cutoff archives and ALFRED vintages; financial source documents "
             + ("were checked against extraction evidence and source hashes." if strict_backtest else "still require local evidence/source-byte verification before an unqualified point-in-time claim."))
            if vintage_dir is not None else "Publication delays and historical revisions remain unresolved in current-vintage exploratory mode.",
            "Both external series are market proxies, not Delta passenger counts or realized ticket yields; coefficients are associations, not causal elasticities.",
            "A small post-2021 sample and overlapping evaluation windows limit error and interval-coverage conclusions.",
            "Ranges use in-sample residuals and imposed horizon scaling, not validated 80% predictive coverage.",
            "The next six filing-based quarters still require an earnings-announcement check before submission.",
            "No LLM signal decisions, human sign-off or automatic publication is performed by this numerical model.",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".drivers_") as temporary:
        staging = Path(temporary)
        forecast.to_csv(staging / "driver_forecast.csv", index=False)
        detail.to_csv(staging / "driver_backtest.csv", index=False)
        comparison.to_csv(staging / "model_comparison.csv", index=False)
        sensitivity_table.to_csv(staging / "driver_sensitivities.csv", index=False)
        training.to_csv(staging / "driver_training.csv", index=False)
        (staging / "driver_run.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for filename in ("driver_forecast.csv", "driver_backtest.csv", "model_comparison.csv",
                         "driver_sensitivities.csv", "driver_training.csv", "driver_run.json"):
            (staging / filename).replace(output_dir / filename)
    return forecast, detail, comparison


def run_baseline(input_path, output_dir, cutoff, origins=None, origin_count=4):
    if date.fromisoformat(str(cutoff)) > date.today():
        raise ValueError("The cutoff cannot be in the future.")
    financials = load_financials(input_path)
    automatic_origins = origins is None
    if automatic_origins:
        origins = default_origins(financials, cutoff, origin_count)
    if len(origins) < 4:
        raise ValueError("Use at least four backtest origins.")
    forecast = seasonal_naive(financials, cutoff)
    detail, metrics = backtest(financials, origins, cutoff)
    available = available_financials(financials, cutoff)
    origin_audit = []
    for origin in sorted(origins):
        known = available_financials(financials, origin)
        origin_audit.append({"cutoff_date": str(origin), "available_quarters": len(known),
                             "latest_available_quarter": str(known["period"].iloc[-1]),
                             "latest_admitted_filing_date": known["published_date"].max().isoformat()})
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff_date": str(cutoff),
        "cutoff_convention": CUTOFF_CONVENTION, "model": MODEL, "units": "USD millions",
        "input_file": Path(input_path).name,
        "input_csv_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "latest_available_quarter": str(available["period"].iloc[-1]),
        "headline_quarter": forecast.iloc[-1]["quarter"],
        "headline_baseline_usd_millions": float(forecast.iloc[-1]["forecast_usd_millions"]),
        "origin_selection": "Latest fully scored March/June/September/December first-day cutoffs; not selected by error" if automatic_origins else "Explicit user-supplied cutoffs",
        "backtest_origins": origin_audit, "external_signals_used": [],
        "external_signal_backtest_ready": False, "intervals_included": False, "submission_ready": False,
        "limitations": [
            "Baseline benchmark only, not a signal-driven or agent-reviewed forecast.",
            "Financial values are the stored first-filing observations, not a full restatement-vintage history.",
            "Filing-date filtering does not verify earlier earnings announcements or the six truly unreported quarters.",
            "External signal release dates and historical vintages remain unverified; they are not admitted to this backtest.",
            "Backtest windows overlap; four recent origins are a small sample, not independent observations.",
            "No prediction intervals, consensus comparison, human approval or publication workflow yet.",
        ],
        "signal_admission_contract": {
            "tsa": "Require observation date, verified historical availability and revision/archival evidence.",
            "eia": "Require observation date, actual release availability and revision/archival evidence.",
            "airfare": "Require observation month, actual release date and cutoff-appropriate historical vintage; source blanks stay missing.",
            "all": "Fit and transform using cutoff-available data only; never use realized future signals as forecast drivers.",
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".baseline_") as temporary:
        staging = Path(temporary)
        forecast.to_csv(staging / "baseline_forecast.csv", index=False)
        detail.to_csv(staging / "baseline_backtest.csv", index=False)
        metrics.to_csv(staging / "baseline_metrics.csv", index=False)
        (staging / "baseline_run.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for filename in ("baseline_forecast.csv", "baseline_backtest.csv", "baseline_metrics.csv", "baseline_run.json"):
            (staging / filename).replace(output_dir / filename)
    return forecast, detail, metrics


def run_vintage_backtest(input_path, vintage_dir, output_dir, cutoff, origins=None, origin_count=4):
    """Evaluate snapshots offline without loading current signals or creating a live forecast."""
    if date.fromisoformat(str(cutoff)) > date.today():
        raise ValueError("The evaluation cutoff cannot be in the future.")
    financials = load_financials(input_path)
    automatic = origins is None
    origins = default_origins(financials, cutoff, origin_count) if automatic else origins
    if len(origins) < 4:
        raise ValueError("Use at least four common backtest origins.")
    detail, comparison, fits = backtest_driver(financials, None, origins, cutoff, vintage_dir, input_path)
    baseline_detail, baseline_metrics = backtest(financials, origins, cutoff)
    source_verified = all(item["financial_audit"]["source_documents_verified"] for item in fits)
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff_date": str(cutoff),
        "model": DRIVER_MODEL, "units": "USD millions", "evaluation_only": True,
        "financial_cutoff_convention": CUTOFF_CONVENTION,
        "signal_data_basis": SNAPSHOT_BASIS, "historical_signal_evidence_verified": True,
        "current_vintage_fallback_used": False,
        "financial_source_documents_verified": source_verified,
        "strict_point_in_time": source_verified,
        "strict_point_in_time_scope": "Numerical backtest input evidence; not historical LLM decisions or earliest earnings-release verification.",
        "input_csv_sha256": {Path(input_path).name: hashlib.sha256(Path(input_path).read_bytes()).hexdigest()},
        "forecasting_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "origin_selection": "Same automatic calendar origins as baseline; not selected by error" if automatic else "Explicit common cutoffs",
        "backtest_origins": fits, "external_signals_used": ["tsa", "airfare"],
        "submission_ready": False, "approval_status": "not_reviewed",
        "limitations": [
            "Historical signals are verified against preserved pre-cutoff source responses. Captures may precede the cutoff by several days and need not include every revision already available then.",
            "Financial source-byte verification passed." if source_verified else "Financial dates are filtered; extraction evidence/original source-byte verification remains incomplete. Inspect each origin's financial_audit.",
            "Stored original-filing financial values are used, not a full history of management earnings-release/restatement vintages.",
            "The numerical model's fixed default signal actions are evaluated; AI recommendations and human overrides are not backtested.",
            "The sample is small and forecast windows overlap. Ranges are illustrative, not calibrated 80% coverage.",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".vintage_backtest_") as temporary:
        staging = Path(temporary)
        baseline_detail.to_csv(staging / "baseline_backtest.csv", index=False)
        baseline_metrics.to_csv(staging / "baseline_metrics.csv", index=False)
        detail.to_csv(staging / "driver_backtest.csv", index=False)
        comparison.to_csv(staging / "model_comparison.csv", index=False)
        (staging / "driver_run.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in ("baseline_backtest.csv", "baseline_metrics.csv", "driver_backtest.csv", "model_comparison.csv", "driver_run.json"):
            (staging / name).replace(output_dir / name)
    return detail, comparison, audit


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base_dir / "input_data" / "delta_quarterly.csv")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "output_data")
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date.today())
    parser.add_argument("--origins", nargs="+", type=date.fromisoformat)
    parser.add_argument("--origin-count", type=int, default=4)
    parser.add_argument("--model", choices=("baseline", "drivers", "both"), default="both")
    parser.add_argument("--tsa-input", type=Path, default=base_dir / "input_data" / "tsa_daily.csv")
    parser.add_argument("--airfare-input", type=Path, default=base_dir / "input_data" / "airfare_cpi_monthly.csv")
    parser.add_argument("--backtest-basis", choices=("vintages", "current-vintage"), default="vintages",
                        help="Default: require origin-specific evidence. Current-vintage is explicitly exploratory.")
    parser.add_argument("--vintage-dir", type=Path,
                        help="Snapshot root; defaults to backtest_vintages beside the financial input CSV.")
    parser.add_argument("--backtest-only", action="store_true",
                        help="Evaluate historical snapshots only; no current-signal load, live forecast or API calls.")
    args = parser.parse_args()
    try:
        origins = [origin.isoformat() for origin in args.origins] if args.origins else None
        vintage_dir = (args.vintage_dir or args.input.parent / "backtest_vintages") if args.backtest_basis == "vintages" else None
        if args.backtest_only:
            if vintage_dir is None:
                raise ValueError("--backtest-only requires --backtest-basis vintages.")
            detail, comparison, audit = run_vintage_backtest(args.input, vintage_dir, args.output_dir,
                                                            args.cutoff, origins, args.origin_count)
            print(f"Historical-snapshot numerical backtest; evaluation cutoff {args.cutoff}; USD millions")
            print(comparison.loc[comparison["horizon"].isin([1, 6])].to_string(index=False))
            for item in audit["backtest_origins"]:
                signal = item["signal_audit"]
                print(f"Origin {item['cutoff_date']}: TSA through {signal['tsa_latest_observation']} "
                      f"({signal['tsa_age_days_at_cutoff']} days old); airfare vintage {signal['airfare_vintage_as_of_date']}; "
                      f"training {item['fit']['training_quarters']} quarters; financial source bytes verified: "
                      f"{item['financial_audit']['source_documents_verified']}")
            print(f"Historical signal evidence verified: {audit['historical_signal_evidence_verified']}")
            print(f"Financial source documents verified: {audit['financial_source_documents_verified']}")
            print(f"Backtest: {detail['cutoff_date'].nunique()} origins, {len(detail)} scored predictions per model")
            print(f"Outputs: {args.output_dir.resolve()}")
            print("No live forecast, LLM request, approval or publication was performed. Review driver_run.json before updating the UI.")
            return 0
        if args.model in ("baseline", "both"):
            forecast, detail, metrics = run_baseline(args.input, args.output_dir, args.cutoff, origins, args.origin_count)
            print(f"Seasonal-naive baseline; cutoff {args.cutoff}; USD millions; not submission-ready")
            print(forecast[["horizon", "quarter", "forecast_usd_millions", "input_quarter", "reference_type"]].to_string(index=False))
            print(metrics.loc[metrics["horizon"].isin([1, 6])].to_string(index=False))
        if args.model in ("drivers", "both"):
            forecast, detail, metrics = run_driver_model(args.input, args.tsa_input, args.airfare_input, args.output_dir,
                                                        args.cutoff, origins, args.origin_count, vintage_dir)
            basis = "historical-snapshot backtest" if vintage_dir is not None else "CURRENT-VINTAGE exploratory backtest"
            print(f"TSA/airfare challenger; cutoff {args.cutoff}; {basis}")
            print(forecast[["horizon", "quarter", "forecast_usd_millions", "lower_80_usd_millions", "upper_80_usd_millions"]].to_string(index=False))
            print(metrics.loc[metrics["horizon"].isin([1, 6])].to_string(index=False))
            print("Review driver_run.json for clipped assumptions. Ranges are illustrative, not coverage-calibrated.")
    except (OSError, ValueError, KeyError) as error:
        print(f"FORECAST FAILED: {error}")
        return 1
    print(f"Backtest: {detail['cutoff_date'].nunique()} origins, {len(detail)} scored predictions")
    print(f"Outputs: {args.output_dir.resolve()}")
    print("Numerical outputs only. Review the saved audit, evaluation scope and approval status before submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

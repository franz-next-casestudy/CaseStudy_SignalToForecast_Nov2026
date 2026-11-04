"""Retrieve Delta financials, TSA throughput, EIA jet fuel, and FRED airfare CPI.

Run: python data_retrieval.py
Use --download-sec to refresh SEC filings with SEC_NAME and SEC_EMAIL configured.
Use --sec-test to test two SEC downloads without running the CSV pipeline.
CSV output defaults to input_data beside this script. No API keys required.
"""

import argparse
import calendar
import contextlib
import csv
import hashlib
import io
import os
import re
import tempfile
from datetime import date, datetime, timezone
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

METRICS = ("passenger_revenue_usd_millions", "rpm_millions", "asm_millions",
           "passenger_yield_cents", "load_factor_percent")


class PageInspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None
        self.contexts = {}
        self.context = None
        self.field = None
        self.facts = []
        self.fact = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = []
        if tag in ("xbrli:context", "context"):
            self.context = {"id": attributes.get("id"), "members": []}
        if self.context is not None and tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant", "xbrldi:explicitmember"):
            self.field = (tag.split(":")[-1], [])
        if tag == "ix:nonfraction":
            self.fact = {"concept": attributes.get("name", ""),
                         "context_ref": attributes.get("contextref"),
                         "unit": attributes.get("unitref"),
                         "scale": attributes.get("scale", "0"),
                         "sign": attributes.get("sign", ""), "text": []}

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            if self.row is not None:
                self.row.append(" ".join(self.cell))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        if self.field is not None and tag.split(":")[-1] == self.field[0]:
            name, parts = self.field
            value = "".join(parts).strip()
            if name == "explicitmember":
                self.context["members"].append(value)
            else:
                self.context[name] = value
            self.field = None
        if tag in ("xbrli:context", "context") and self.context is not None:
            self.contexts[self.context["id"]] = self.context
            self.context = None
        if tag == "ix:nonfraction" and self.fact is not None:
            self.fact["text"] = "".join(self.fact["text"]).strip()
            self.facts.append(self.fact)
            self.fact = None

    def handle_data(self, data):
        if self.cell is not None and data.strip():
            self.cell.append(data.strip())
        if self.field is not None:
            self.field[1].append(data)
        if self.fact is not None:
            self.fact["text"].append(data)


def clean_text(value):
    return " ".join(value.split())


def number(value):
    text = clean_text(value).replace(",", "").replace("$", "").replace("%", "").strip()
    if not re.fullmatch(r"\(?-?\d+(?:\.\d+)?\)?", text):
        return None
    return float(text.replace("(", "-").replace(")", ""))


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        raise ValueError(f"Refusing to write an empty dataset: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download(dataset, url, output_dir, timeout, log):
    entry = {"dataset": dataset, "requested_url": url, "source_url": "",
             "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
             "http_status": "", "sha256": "", "raw_file": "", "status": "failed", "error": ""}
    log.append(entry)
    try:
        request = Request(url, headers={"User-Agent": "SignalToForecastCaseStudy/1.0"})
        if dataset == "fred_airfare_cpi":
            import requests

            try:
                response = requests.get(url, timeout=timeout)
                entry.update(source_url=response.url, http_status=response.status_code)
                response.raise_for_status()
                body = response.content
            except requests.RequestException as error:
                raise OSError(f"FRED CSV download failed: {error}") from error
        else:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                entry.update(source_url=response.geturl(), http_status=response.status)
        entry["sha256"] = hashlib.sha256(body).hexdigest()
        suffix = ".xls" if body.startswith(bytes.fromhex("D0CF11E0")) else ".html"
        if url.split("?", 1)[0].endswith(".csv"):
            suffix = ".csv"
        snapshot = output_dir / "raw" / f"{dataset}_{entry['sha256'][:16]}{suffix}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(body)
        entry.update(raw_file=str(snapshot), status="downloaded")
        return body, entry
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        entry["error"] = str(error)
        if isinstance(error, HTTPError):
            entry.update(http_status=error.code, source_url=error.geturl())
        raise


def download_filings(downloader, form, after, before, limit, output_dir):
    form_dir = output_dir / "sec-edgar-filings" / "0000027904" / form
    before_files = set(form_dir.rglob("full-submission.txt")) if form_dir.exists() else set()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            count = downloader.get(form, "0000027904", after=after, before=before,
                                   limit=limit, download_details=True)
    except Exception as error:
        print(f"{form}: download failed ({type(error).__name__}). No identity values were logged.")
        return False
    after_files = set(form_dir.rglob("full-submission.txt")) if form_dir.exists() else set()
    print(f"{form}: downloader reported {count} filings; {len(after_files - before_files)} new submission files.")
    return bool(count) and bool(after_files)


def write_manifest(output_dir):
    rows = []
    checked_at = datetime.now(timezone.utc).isoformat()
    for path in sorted((output_dir / "sec-edgar-filings").rglob("full-submission.txt")):
        body = path.read_bytes()
        accession = path.parent.name
        rows.append({"form": path.parent.parent.name, "accession_number": accession,
                     "local_file": str(path.relative_to(output_dir)),
                     "source_url": f"https://www.sec.gov/Archives/edgar/data/27904/{accession}.txt",
                     "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                     "inventory_checked_at_utc": checked_at})
    write_csv(output_dir / "filing_manifest.csv", rows,
              ["form", "accession_number", "local_file", "source_url", "bytes", "sha256", "inventory_checked_at_utc"])


def refresh_sec(sec_dir, start_year, end_year, smoke_test=False):
    name = os.environ.get("SEC_NAME", "").strip()
    email = os.environ.get("SEC_EMAIL", "").strip()
    if not name or not email:
        raise ValueError("Set SEC_NAME and SEC_EMAIL in the same terminal that runs this script.")
    if "@" not in email or any(character in name + email for character in "\r\n"):
        raise ValueError("SEC identity configuration is invalid; check the email and remove line breaks.")
    from sec_edgar_downloader import Downloader

    sec_dir.mkdir(parents=True, exist_ok=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            downloader = Downloader(name, email, sec_dir)
    except Exception as error:
        raise ValueError(f"SEC downloader initialization failed ({type(error).__name__}); identity values not logged.") from None
    after = f"{start_year}-01-01"
    before = min(date.today(), date(end_year + 1, 12, 31)).isoformat()
    checks = [(form, after, before, None) for form in ("10-Q", "10-K")]
    if smoke_test:
        checks = [("10-Q", after, before, 1), ("10-K", "2020-01-01", "2020-12-31", 1)]
    successful = True
    for form, after, before, limit in checks:
        if not download_filings(downloader, form, after, before, limit, sec_dir):
            successful = False
    write_manifest(sec_dir)
    if not successful:
        raise ValueError("SEC access was not confirmed for every request. Do not retry repeatedly if blocked.")
    print("SEC documents inventoried. Existing cached files do not prove a new document download.")


def operating_facts(inspector, report_end, provenance):
    observations = []
    for index, cells in enumerate(inspector.rows):
        if not cells:
            continue
        label = clean_text(cells[0]).lower()
        metric = next((name for name, prefix in (("rpm_millions", "revenue passenger miles"),
                                                ("asm_millions", "available seat miles"),
                                                ("passenger_yield_cents", "passenger mile yield"),
                                                ("load_factor_percent", "passenger load factor"))
                       if label.startswith(prefix)), None)
        if metric is None:
            continue
        headings = [previous for previous in range(max(0, index - 25), index)
                    if inspector.rows[previous] and clean_text(inspector.rows[previous][0]).lower().startswith("consolidated")]
        if not headings:
            continue
        heading = headings[-1]
        years = [number(cell) for cell in inspector.rows[heading][1:] if number(cell) is not None]
        period_header = " ".join(clean_text(" ".join(row)) for row in inspector.rows[max(0, heading - 2):heading])
        if not years or years[0] != report_end.year:
            continue
        if "Three Months Ended" in period_header and provenance["form"] == "10-Q":
            start = date(report_end.year, ((report_end.month - 1) // 3) * 3 + 1, 1)
        elif "Year Ended December" in period_header and provenance["form"] == "10-K":
            start = date(report_end.year, 1, 1)
        else:
            continue
        if calendar.month_name[report_end.month].lower() not in period_header.lower():
            continue
        values = [number(cell) for cell in cells[1:] if number(cell) is not None]
        if not values:
            continue
        if metric in ("rpm_millions", "asm_millions") and "millions" not in label:
            raise ValueError(f"Unverified traffic units in {provenance['accession_number']}")
        if metric == "passenger_yield_cents" and "\u00a2" not in " ".join(cells) and "cents" not in label:
            raise ValueError(f"Unverified yield units in {provenance['accession_number']}")
        unit = "million passenger miles" if metric == "rpm_millions" else (
            "million seat miles" if metric == "asm_millions" else ("cents per passenger mile" if metric == "passenger_yield_cents" else "percent"))
        observations.append({"start": start.isoformat(), "end": report_end.isoformat(),
                             "value": Decimal(str(values[0])), "unit": unit, "metric": metric,
                             "concept": cells[0], "context_ref": f"HTML row {index + 1}: {period_header}",
                             "members": "Consolidated", "method": "reported_table",
                             **provenance})
    return observations


def header_value(header, label):
    match = re.search(rf"^{re.escape(label)}:\s*(\S+)", header, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing SEC header field: {label}")
    return match[1]


def inspect_filing(manifest, sec_dir):
    submission = sec_dir / manifest["local_file"].replace("\\", "/")
    with submission.open(encoding="utf-8", errors="replace") as handle:
        prefix = handle.read(65536)
    header = prefix.split("</SEC-HEADER>", 1)[0]
    report_end = datetime.strptime(header_value(header, "CONFORMED PERIOD OF REPORT"), "%Y%m%d").date()
    filed = datetime.strptime(header_value(header, "FILED AS OF DATE"), "%Y%m%d").date()
    acceptance = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", header)
    primary = submission.with_name("primary-document.html")
    body = primary.read_bytes()
    inspector = PageInspector()
    inspector.feed(body.decode("utf-8", errors="replace"))
    filename = re.search(r"<FILENAME>([^\r\n]+)", prefix)
    accession = manifest["accession_number"]
    source_url = (f"https://www.sec.gov/Archives/edgar/data/27904/{accession.replace('-', '')}/{filename[1].strip()}"
                  if filename else manifest["source_url"])
    provenance = {"accession_number": accession, "form": manifest["form"],
                  "filed_date": filed.isoformat(),
                  "accepted_datetime_et": acceptance[1] if acceptance else "",
                  "source_url": source_url, "submission_url": manifest["source_url"],
                  "local_file": str(primary.relative_to(sec_dir)),
                  "source_sha256": hashlib.sha256(body).hexdigest()}
    observations = []
    for fact in inspector.facts:
        context = inspector.contexts.get(fact["context_ref"], {})
        if fact["concept"] != "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax":
            continue
        if context.get("members") != ["us-gaap:PassengerMember"]:
            continue
        if not context.get("startdate") or not context.get("enddate"):
            continue
        start, end = date.fromisoformat(context["startdate"]), date.fromisoformat(context["enddate"])
        if start.year != report_end.year or end > report_end:
            continue
        if fact["unit"].lower() != "usd":
            raise ValueError(f"Unexpected passenger revenue unit in {accession}: {fact['unit']}")
        value = Decimal(fact["text"].replace(",", "")) * Decimal(10) ** int(fact["scale"])
        if fact["sign"] == "-":
            value = -value
        observations.append({"start": start.isoformat(), "end": end.isoformat(),
                             "value": value / Decimal(1000000), "unit": "USD millions",
                             "metric": "passenger_revenue_usd_millions", "concept": fact["concept"],
                             "context_ref": fact["context_ref"], "members": "us-gaap:PassengerMember",
                             "method": "reported_xbrl", **provenance})
    if not observations:
        raise ValueError(f"No current-year passenger revenue facts found in {accession}")
    observations.extend(operating_facts(inspector, report_end, provenance))
    unique = {}
    for observation in observations:
        key = (observation["metric"], observation["start"], observation["end"])
        if key in unique and unique[key]["value"] != observation["value"]:
            raise ValueError(f"Conflicting revenue facts in {accession}: {key}")
        unique[key] = observation
    return report_end, list(unique.values())


def build_quarters(facts, start_year):
    rows, evidence = [], []
    revenue_facts = [fact for fact in facts if fact["metric"] == "passenger_revenue_usd_millions"]
    latest_end = max(date.fromisoformat(fact["end"]) for fact in facts)
    for year in range(start_year, latest_end.year + 1):
        for quarter in range(1, 5):
            end_month = quarter * 3
            end = date(year, end_month, calendar.monthrange(year, end_month)[1])
            if end > latest_end:
                continue
            start = date(year, end_month - 2, 1)
            quarter_name = f"{year}Q{quarter}"
            reconciliation_difference = ""
            direct = [fact for fact in revenue_facts if fact["start"] == start.isoformat() and fact["end"] == end.isoformat()]
            if direct:
                chosen = min(direct, key=lambda fact: (fact["filed_date"], fact["accession_number"]))
                value, inputs, method = chosen["value"], [chosen], "reported_xbrl"
            elif quarter == 4:
                annual = [fact for fact in revenue_facts if fact["start"] == f"{year}-01-01" and fact["end"] == end.isoformat()]
                if not annual:
                    raise ValueError(f"Missing annual passenger revenue for {quarter_name}")
                chosen = min(annual, key=lambda fact: (fact["filed_date"], fact["accession_number"]))
                ytd = [fact for fact in revenue_facts if fact["start"] == f"{year}-01-01" and fact["end"] == f"{year}-09-30"
                       and fact["filed_date"] <= chosen["filed_date"]]
                if not ytd:
                    raise ValueError(f"Missing nine-month passenger revenue for {quarter_name}")
                nine_month = max(ytd, key=lambda fact: (fact["filed_date"], fact["accession_number"]))
                value, inputs, method = chosen["value"] - nine_month["value"], [chosen, nine_month], "annual_minus_nine_month"
                prior = [row for row in rows if row["quarter"].startswith(str(year))]
                reconciliation_difference = nine_month["value"] - sum(Decimal(row["passenger_revenue_usd_millions"]) for row in prior)
                if abs(reconciliation_difference) > Decimal(2):
                    raise ValueError(f"Quarterly revenue does not reconcile to nine-month revenue for {year}")
            else:
                raise ValueError(f"Missing directly reported passenger revenue for {quarter_name}")
            if value <= 0:
                raise ValueError(f"Nonpositive passenger revenue in {quarter_name}")
            row = {"quarter": quarter_name, "period_end": end.isoformat(),
                   "published_date": max(fact["filed_date"] for fact in inputs),
                   **{metric: "" for metric in METRICS}, "revenue_method": method,
                   "nine_month_minus_quarter_sum_usd_millions": str(reconciliation_difference),
                   "source_url": chosen["source_url"],
                   "additional_source_url": inputs[1]["source_url"] if len(inputs) == 2 else "",
                   "accession_number": chosen["accession_number"], "source_sha256": chosen["source_sha256"],
                   "availability_basis": "SEC filing date; earnings may have been public earlier"}
            row["passenger_revenue_usd_millions"] = str(value)
            for index, fact in enumerate(inputs):
                evidence.append({"quarter": quarter_name, "input_role": "reported" if len(inputs) == 1 else ("annual" if index == 0 else "nine_month"),
                                 "result_metric": "passenger_revenue_usd_millions", "result_value": str(value),
                                 "formula": "annual - nine_month" if quarter == 4 else "reported",
                                 **{key: str(value) if key == "value" else value for key, value in fact.items()}})
            for metric in ("rpm_millions", "asm_millions", "passenger_yield_cents", "load_factor_percent"):
                candidates = [fact for fact in facts if fact["metric"] == metric and fact["start"] == start.isoformat()
                              and fact["end"] == end.isoformat()]
                if candidates:
                    statistic = min(candidates, key=lambda fact: (fact["filed_date"], fact["accession_number"]))
                    row[metric] = str(statistic["value"])
                    evidence.append({"quarter": quarter_name, "input_role": "reported", "result_metric": metric,
                                     "result_value": str(statistic["value"]), "formula": "reported",
                                     **{key: str(value) if key == "value" else value for key, value in statistic.items()}})
                elif quarter == 4 and metric in ("rpm_millions", "asm_millions"):
                    annual_stats = [fact for fact in facts if fact["metric"] == metric and fact["start"] == f"{year}-01-01"
                                    and fact["end"] == end.isoformat() and fact["filed_date"] <= row["published_date"]]
                    prior = [previous for previous in rows if previous["quarter"].startswith(str(year))]
                    if annual_stats and len(prior) == 3 and all(previous[metric] != "" for previous in prior):
                        statistic = min(annual_stats, key=lambda fact: (fact["filed_date"], fact["accession_number"]))
                        row[metric] = str(statistic["value"] - sum(Decimal(previous[metric]) for previous in prior))
                        stat_inputs = [statistic] + [fact for previous in prior for fact in evidence
                                                   if fact["quarter"] == previous["quarter"] and fact["result_metric"] == metric
                                                   and fact["input_role"] == "reported"]
                        for stat_index, stat_input in enumerate(stat_inputs):
                            evidence.append({**stat_input, "quarter": quarter_name,
                                             "input_role": "annual" if stat_index == 0 else "prior_quarter",
                                             "result_metric": metric, "result_value": row[metric],
                                             "formula": "annual - Q1 - Q2 - Q3",
                                             "value": str(stat_input["value"])})
            if quarter == 4:
                for metric, numerator, denominator in (("passenger_yield_cents", "passenger_revenue_usd_millions", "rpm_millions"),
                                                       ("load_factor_percent", "rpm_millions", "asm_millions")):
                    if row[numerator] != "" and row[denominator] != "":
                        row[metric] = str((100 * Decimal(row[numerator]) / Decimal(row[denominator])).quantize(Decimal("0.0001")))
                        ratio_inputs = [fact for fact in evidence if fact["quarter"] == quarter_name
                                        and fact["result_metric"] in (numerator, denominator)]
                        for ratio_input in ratio_inputs:
                            evidence.append({**ratio_input, "input_role": "ratio_input", "result_metric": metric,
                                             "result_value": row[metric], "formula": f"100 * {numerator} / {denominator}"})
            row["missing_operating_metrics"] = ";".join(metric for metric in METRICS if row[metric] == "")
            rows.append(row)
    return rows, evidence


def extract_dataset(sec_dir, start_year=2019, end_year=None):
    with (sec_dir / "filing_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    facts = []
    for filing in manifest:
        report_end, observations = inspect_filing(filing, sec_dir)
        if report_end.year >= start_year and (end_year is None or report_end.year <= end_year):
            facts.extend(observations)
        print(f"{filing['form']} {report_end}: {len(observations)} metric-period facts")
    if not facts:
        raise ValueError("No facts were found in the selected years")
    rows, evidence = build_quarters(facts, start_year)
    validate_dataset(rows, evidence)
    return rows, evidence


def validate_dataset(rows, evidence):
    if not rows or len({row["quarter"] for row in rows}) != len(rows):
        raise ValueError("Missing or duplicate quarters")
    for row in rows:
        for metric in METRICS:
            if row[metric] == "" or Decimal(row[metric]) <= 0:
                raise ValueError(f"Missing or nonpositive {metric} in {row['quarter']}")
            if not any(item["quarter"] == row["quarter"] and item["result_metric"] == metric
                       and item["source_url"].startswith("https://www.sec.gov/Archives/") for item in evidence):
                raise ValueError(f"Missing provenance for {metric} in {row['quarter']}")
        if row["published_date"] <= row["period_end"]:
            raise ValueError(f"Invalid filing date for {row['quarter']}")
        rpm, asm = Decimal(row["rpm_millions"]), Decimal(row["asm_millions"])
        revenue, yield_cents = Decimal(row["passenger_revenue_usd_millions"]), Decimal(row["passenger_yield_cents"])
        load_factor = Decimal(row["load_factor_percent"])
        if not 0 < load_factor <= 100 or rpm > asm:
            raise ValueError(f"Invalid traffic/load factor in {row['quarter']}")
        if abs(revenue - rpm * yield_cents / 100) > rpm * Decimal("0.00005") + 1:
            raise ValueError(f"Revenue/RPM/yield identity mismatch in {row['quarter']}")
        if abs(100 * rpm / asm - load_factor) > Decimal("0.55"):
            raise ValueError(f"RPM/ASM/load factor identity mismatch in {row['quarter']}")


def parse_tsa(inspector):
    observations = {}
    for cells in inspector.rows:
        if len(cells) < 2:
            continue
        try:
            observation_date = datetime.strptime(cells[0], "%m/%d/%Y").date()
            passengers = int(cells[1].replace(",", "").replace(" ", ""))
        except ValueError:
            continue
        if passengers > 0 and observation_date <= date.today():
            observations[observation_date.isoformat()] = passengers
    return [{"date": day, "passengers": str(count)} for day, count in sorted(observations.items())]


def parse_eia(body):
    import xlrd

    workbook = xlrd.open_workbook(file_contents=body)
    observations = []
    for sheet in workbook.sheets():
        for row_index in range(sheet.nrows):
            if sheet.ncols < 2:
                continue
            date_cell, price_cell = sheet.cell(row_index, 0), sheet.cell(row_index, 1)
            if date_cell.ctype != xlrd.XL_CELL_DATE or price_cell.ctype != xlrd.XL_CELL_NUMBER:
                continue
            day = xlrd.xldate_as_datetime(date_cell.value, workbook.datemode).date()
            if day <= date.today() and price_cell.value > 0:
                observations.append({"date": day.isoformat(), "usd_per_gallon": str(price_cell.value)})
    return observations


def metadata(entry):
    return {"published_date": "", "availability_status": "historical_publication_and_revisions_not_verified",
            "source_url": entry["source_url"], "retrieved_at_utc": entry["retrieved_at_utc"],
            "source_sha256": entry["sha256"]}


def retrieve_tsa(start_year, end_year, output_dir, timeout, log):
    observations = []
    for year in range(start_year, end_year + 1):
        url = "https://www.tsa.gov/travel/passenger-volumes"
        if year != date.today().year:
            url += f"/{year}"
        body, entry = download(f"tsa_{year}", url, output_dir, timeout, log)
        inspector = PageInspector()
        inspector.feed(body.decode("utf-8", errors="replace"))
        rows = parse_tsa(inspector)
        if not rows:
            raise ValueError(f"TSA {year}: no daily observations found.")
        if any(date.fromisoformat(row["date"]).year != year for row in rows):
            raise ValueError(f"TSA {year}: unexpected observation year.")
        if len({row["date"] for row in rows}) != len(rows):
            raise ValueError(f"TSA {year}: duplicate dates.")
        expected_end = date(year, 12, 31) if year < date.today().year else date.fromisoformat(rows[-1]["date"])
        expected_days = (expected_end - date(year, 1, 1)).days + 1
        if len(rows) != expected_days:
            raise ValueError(f"TSA {year}: expected {expected_days} daily observations, found {len(rows)}.")
        if year == date.today().year and (date.today() - expected_end).days > 7:
            raise ValueError("Current TSA observations are more than seven days old.")
        observations.extend({**row, **metadata(entry)} for row in rows)
        entry["status"] = "parsed"
        print(f"TSA {year}: {len(rows)} daily observations")
    return sorted(observations, key=lambda row: row["date"])


def retrieve_fuel(start_year, end_year, output_dir, timeout, log):
    url = "https://www.eia.gov/dnav/pet/hist_xls/EER_EPJK_PF4_RGC_DPGw.xls"
    body, entry = download("eia_jet_fuel", url, output_dir, timeout, log)
    rows = [row for row in parse_eia(body) if start_year <= date.fromisoformat(row["date"]).year <= end_year]
    if not rows or len({row["date"] for row in rows}) != len(rows):
        raise ValueError("EIA observations are missing or contain duplicate dates.")
    rows.sort(key=lambda row: row["date"])
    if end_year == date.today().year and (date.today() - date.fromisoformat(rows[-1]["date"])).days > 14:
        raise ValueError("Latest EIA observation is more than fourteen days old.")
    entry["status"] = "parsed"
    print(f"EIA: {len(rows)} weekly observations, {rows[0]['date']} through {rows[-1]['date']}")
    return [{**row, **metadata(entry)} for row in rows]


def parse_airfare(body, start_year, end_year):
    series_id = "CUSR0000SETG01"
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    date_column = next((name for name in ("observation_date", "DATE") if name in fields), None)
    if date_column is None or series_id not in fields or len(fields) != 2:
        raise ValueError("FRED response does not contain the expected airfare CSV columns.")
    observations = {}
    for source_row in reader:
        day = date.fromisoformat(source_row[date_column])
        if not start_year <= day.year <= end_year:
            continue
        if day.day != 1 or day > date.today() or day.isoformat() in observations:
            raise ValueError("FRED airfare CSV contains an invalid or duplicate observation month.")
        raw_value = source_row[series_id].strip()
        missing = raw_value in ("", ".")
        if not missing:
            value = Decimal(raw_value)
            if not value.is_finite() or value <= 0:
                raise ValueError(f"Invalid airfare CPI value for {day}")
        observations[day.isoformat()] = {
            "date": day.isoformat(), "airfare_cpi_index": "" if missing else raw_value,
            "value_status": "missing_at_source" if missing else "reported"}
    rows = [observations[day] for day in sorted(observations)]
    if not rows or not any(row["value_status"] == "reported" for row in rows):
        raise ValueError("No valid airfare CPI observations found in the selected years.")
    latest = date.fromisoformat(rows[-1]["date"])
    expected_count = (latest.year - start_year) * 12 + latest.month
    if len(rows) != expected_count or rows[0]["date"] != f"{start_year}-01-01":
        raise ValueError("FRED airfare CSV has missing monthly rows; source blanks are retained separately.")
    if end_year < date.today().year and latest != date(end_year, 12, 1):
        raise ValueError("FRED airfare history ends before the selected historical end year.")
    latest_reported = max(date.fromisoformat(row["date"]) for row in rows if row["value_status"] == "reported")
    if end_year == date.today().year and (date.today() - latest_reported).days > 95:
        raise ValueError("Latest airfare CPI observation month is more than 95 days old.")
    return rows


def retrieve_airfare(start_year, end_year, output_dir, timeout, log):
    series_id = "CUSR0000SETG01"
    end_date = min(date.today(), date(end_year, 12, 31)).isoformat()
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd={start_year}-01-01&coed={end_date}")
    body, entry = download("fred_airfare_cpi", url, output_dir, timeout, log)
    rows = parse_airfare(body, start_year, end_year)
    entry["status"] = "parsed"
    missing_count = sum(row["value_status"] == "missing_at_source" for row in rows)
    print(f"FRED airfare CPI: {len(rows)} monthly rows, {rows[0]['date']} through {rows[-1]['date']}; {missing_count} source blanks retained")
    return [{**row, **metadata(entry),
             "availability_status": "latest_vintage_not_point_in_time; historical_publication_not_verified",
             "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
             "download_url": entry["source_url"], "series_id": series_id,
             "units": "Index 1982-1984=100", "seasonal_adjustment": "Seasonally Adjusted",
             "frequency": "Monthly", "data_source": "US Bureau of Labor Statistics via FRED"}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sec_mode = parser.add_mutually_exclusive_group()
    sec_mode.add_argument("--download-sec", action="store_true", help="Refresh SEC filings before generating CSVs; requires SEC_NAME and SEC_EMAIL.")
    sec_mode.add_argument("--sec-test", action="store_true", help="Test one recent 10-Q and one older 10-K download, then stop.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "input_data")
    parser.add_argument("--timeout", type=float, default=20, help="Timeout in seconds for each request.")
    parser.add_argument("--sources", nargs="+", choices=("delta", "tsa", "eia", "airfare"), default=["delta", "tsa", "eia", "airfare"])
    args = parser.parse_args()
    if not 2019 <= args.start_year <= args.end_year <= date.today().year or args.timeout <= 0:
        parser.error("Use years from 2019 through the current year and a positive timeout.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    try:
        if args.download_sec or args.sec_test:
            refresh_sec(args.output_dir / "sec", args.start_year, args.end_year, args.sec_test)
            if args.sec_test:
                return 0
        with tempfile.TemporaryDirectory(dir=args.output_dir, prefix=".retrieval_") as temporary:
            staging = Path(temporary)
            if "delta" in args.sources:
                rows, evidence = extract_dataset(args.output_dir / "sec", args.start_year, args.end_year)
                write_csv(staging / "delta_quarterly.csv", rows)
                evidence_fields = list(dict.fromkeys(key for entry in evidence for key in entry))
                write_csv(staging / "delta_evidence.csv", evidence, evidence_fields)
            if "tsa" in args.sources:
                rows = retrieve_tsa(args.start_year, args.end_year, args.output_dir, args.timeout, log)
                write_csv(staging / "tsa_daily.csv", rows)
            if "eia" in args.sources:
                rows = retrieve_fuel(args.start_year, args.end_year, args.output_dir, args.timeout, log)
                write_csv(staging / "jet_fuel_weekly.csv", rows)
            if "airfare" in args.sources:
                rows = retrieve_airfare(args.start_year, args.end_year, args.output_dir, args.timeout, log)
                write_csv(staging / "airfare_cpi_monthly.csv", rows)
            for filename in ("delta_quarterly.csv", "delta_evidence.csv", "tsa_daily.csv", "jet_fuel_weekly.csv", "airfare_cpi_monthly.csv"):
                source = staging / filename
                if source.exists():
                    source.replace(args.output_dir / filename)
        print(f"CSV datasets saved in {args.output_dir.resolve()}")
        print("Historical signal publication dates and revisions remain unverified; this is not yet a point-in-time backtest dataset.")
        return 0
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, ImportError, ArithmeticError, KeyError) as error:
        if log and log[-1]["status"] == "downloaded":
            log[-1].update(status="parse_failed", error=str(error))
        print(f"RETRIEVAL FAILED: {error}")
        print("Existing complete datasets were not replaced. See retrieval_log.csv for source diagnostics.")
        return 1
    finally:
        if log:
            write_csv(args.output_dir / "retrieval_log.csv", log)


if __name__ == "__main__":
    raise SystemExit(main())
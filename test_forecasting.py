"""Focused tests for the filing-date-gated baseline and historical scoring."""

import json
import hashlib
import tempfile
import unittest
import shutil
import sys
from urllib.error import URLError
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from forecast_workflow import (approve_draft, atomic_json, build_draft, latest_saved_forecast, load_record, prepare_context,
                               require_current, save_draft, tsa_freshness, validate_decisions,
                               attach_narrative, narrative_context, seal_record)
from forecast_workflow import financial_evaluation_match, financial_model_reference
from forecast_workflow import load_evaluation_snapshot
from annotate_backtest_financials import annotate
from run_controls import consume_attempt, attempt_usage
from data_retrieval import (TSA_FALLBACK_URL, daily_dates, main as retrieval_main,
                            parse_tsa_fallback, retrieve_tsa, retrieve_tsa_official_year, write_csv)

from forecasting import (REVENUE, available_financials, backtest, default_origins,
                         add_driver_ranges, backtest_driver, driver_forecast, fit_driver,
                         load_financials, load_signals, quarterly_signals, run_baseline,
                         run_driver_model, seasonal_naive)


class TsaFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.entry = {"source_url": TSA_FALLBACK_URL, "retrieved_at_utc": "2026-11-04T00:00:00+00:00", "sha256": "b" * 64}
        self.payload = {"window": {"from": "2026-10-16", "to": "2026-10-17"},
                        "daily": [{"date": "2026-10-16", "people": 1, "tsa_checkpoint_total": 2000000},
                                  {"date": "2026-10-17", "people": 2, "tsa_checkpoint_total": 2100000}]}

    def saved_row(self, day, passengers):
        return {"date": day, "passengers": str(passengers), "published_date": "",
                "availability_status": "historical_publication_and_revisions_not_verified",
                "source_url": "https://www.tsa.gov/travel/passenger-volumes", "source_sha256": "a" * 64,
                "retrieved_at_utc": "2026-11-04T00:00:00+00:00"}

    def test_only_checkpoint_totals_are_used(self):
        rows = parse_tsa_fallback(json.dumps(self.payload).encode())
        self.assertEqual([row["passengers"] for row in rows], ["2000000", "2100000"])
        invalid = json.loads(json.dumps(self.payload))
        del invalid["daily"][0]["tsa_checkpoint_total"]
        with self.assertRaisesRegex(ValueError, "people is not a substitute"):
            parse_tsa_fallback(json.dumps(invalid).encode())

    def test_official_parser_checks_coverage_and_does_not_reject_age_before_merge(self):
        payload = b"<table><tr><td>01/01/2026</td><td>2,000,000</td></tr><tr><td>01/02/2026</td><td>2,100,000</td></tr></table>"
        entry = {**self.entry, "source_url": "https://www.tsa.gov/travel/passenger-volumes"}
        with patch("data_retrieval.download", return_value=(payload, entry)):
            rows = retrieve_tsa_official_year(2026, self.directory, 20, [])
        self.assertEqual(rows[-1]["date"], "2026-01-02")
        self.assertEqual(entry["status"], "parsed")
        duplicate = payload.replace(b"01/02/2026", b"01/01/2026")
        entry = dict(entry)
        with patch("data_retrieval.download", return_value=(duplicate, entry)), self.assertRaisesRegex(ValueError, "duplicate"):
            retrieve_tsa_official_year(2026, self.directory, 20, [])
        self.assertEqual(entry["status"], "parse_failed")

    def test_fallback_fills_missing_dates_without_replacing_saved_values(self):
        saved = [self.saved_row("2026-10-16", 1999999), self.saved_row("2026-11-03", 1966283)]
        write_csv(self.directory / "tsa_daily.csv", saved)
        log = []
        def fallback_download(*args):
            entry = {**self.entry}
            log.append(entry)
            return json.dumps(self.payload).encode(), entry
        with patch("data_retrieval.retrieve_tsa_official_year", side_effect=URLError("blocked")), patch("data_retrieval.download", side_effect=fallback_download):
            rows = retrieve_tsa(2019, 2026, self.directory, 20, log)
        mapped = {row["date"]: row for row in rows}
        self.assertEqual(mapped["2026-10-16"], saved[0])
        self.assertEqual(mapped["2026-11-03"], saved[1])
        self.assertEqual(mapped["2026-10-17"]["source_url"], TSA_FALLBACK_URL)
        self.assertEqual(mapped["2026-10-17"]["source_sha256"], "b" * 64)
        self.assertIn("fallback_security", mapped["2026-10-17"]["availability_status"])
        self.assertEqual(log[-1]["status"], "degraded")
        self.assertEqual(json.loads(log[-1]["details_json"])["fallback_rows_added"], 1)

    def test_no_existing_data_keeps_true_fallback_coverage(self):
        entry = dict(self.entry)
        with patch("data_retrieval.retrieve_tsa_official_year", side_effect=URLError("blocked")), patch("data_retrieval.download", return_value=(json.dumps(self.payload).encode(), entry)):
            rows = retrieve_tsa(2019, 2026, self.directory, 20, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["date"], "2026-10-17")
        self.assertEqual(entry["status"], "parsed_fallback")

    def test_primary_success_does_not_call_fallback_or_drop_newer_saved_rows(self):
        saved = [self.saved_row("2026-11-03", 1966283)]
        write_csv(self.directory / "tsa_daily.csv", saved)
        official = [self.saved_row("2026-11-02", 2444382)]
        with patch("data_retrieval.retrieve_tsa_official_year", return_value=official), patch("data_retrieval.download") as download:
            rows = retrieve_tsa(2026, 2026, self.directory, 20, [])
        download.assert_not_called()
        self.assertEqual(rows[-1], saved[0])

    def test_invalid_fallback_or_cache_never_silently_succeeds(self):
        for count in (None, True, 0, -1, 2000000.5):
            invalid = json.loads(json.dumps(self.payload))
            invalid["daily"][0]["tsa_checkpoint_total"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                parse_tsa_fallback(json.dumps(invalid).encode())
        for daily in ([self.payload["daily"][0]], [self.payload["daily"][0]] * 2):
            invalid = {**self.payload, "daily": daily}
            with self.assertRaises(ValueError):
                parse_tsa_fallback(json.dumps(invalid).encode())
        saved = [self.saved_row("2026-11-03", 1966283)]
        saved[0]["source_sha256"] = "bad"
        write_csv(self.directory / "tsa_daily.csv", saved)
        with patch("data_retrieval.retrieve_tsa_official_year") as official, self.assertRaisesRegex(ValueError, "hash"):
            retrieve_tsa(2019, 2026, self.directory, 20, [])
        official.assert_not_called()

    def test_both_sources_fail_retains_valid_cache_and_attempts_once(self):
        saved = [self.saved_row("2026-11-02", 2444382), self.saved_row("2026-11-03", 1966283)]
        write_csv(self.directory / "tsa_daily.csv", saved)
        before = (self.directory / "tsa_daily.csv").read_bytes()
        log = []
        with patch("data_retrieval.retrieve_tsa_official_year", side_effect=URLError("blocked")) as official, patch("data_retrieval.download", side_effect=OSError("Backup unavailable")) as backup:
            rows = retrieve_tsa(2019, 2026, self.directory, 20, log)
        self.assertEqual(rows, saved)
        self.assertEqual(official.call_count, 1)
        self.assertEqual(backup.call_count, 1)
        self.assertEqual((self.directory / "tsa_daily.csv").read_bytes(), before)
        details = json.loads(log[-1]["details_json"])
        self.assertEqual(details["recovery"], "cache_only")
        self.assertFalse(details["stale"])
        self.assertGreater(details["missing_count"], 0)
        with patch("data_retrieval.retrieve_tsa_official_year", side_effect=URLError("blocked")), patch("data_retrieval.download", side_effect=OSError("Backup unavailable")):
            with self.assertRaisesRegex(ValueError, "No validated"):
                retrieve_tsa(2019, 2026, self.directory / "empty", 20, [])

    def test_complete_historical_year_is_reused_and_current_year_is_requested(self):
        historical = [self.saved_row(day, 2000000) for day in sorted(daily_dates(date(2025, 1, 1), date(2025, 12, 31)))]
        latest = self.saved_row("2026-11-03", 1966283)
        write_csv(self.directory / "tsa_daily.csv", historical + [latest])
        official_rows = [self.saved_row(day, 2100000) for day in sorted(daily_dates(date(2026, 1, 1), date(2026, 11, 2)))]
        log = []
        with patch("data_retrieval.retrieve_tsa_official_year", return_value=official_rows) as official, patch("data_retrieval.download") as backup:
            rows = retrieve_tsa(2025, 2026, self.directory, 20, log)
        self.assertEqual([call.args[0] for call in official.call_args_list], [2026])
        backup.assert_not_called()
        self.assertEqual(rows[-1], latest)
        self.assertEqual(json.loads(log[-1]["details_json"])["historical_years_reused"], [2025])
        with patch("data_retrieval.retrieve_tsa_official_year", side_effect=lambda year, *args: historical if year == 2025 else official_rows) as official:
            retrieve_tsa(2025, 2026, self.directory, 20, [], refresh_history=True)
        self.assertEqual([call.args[0] for call in official.call_args_list], [2025, 2026])

    def test_strict_checks_reject_stale_or_incomplete_coverage(self):
        write_csv(self.directory / "tsa_daily.csv", [self.saved_row("2026-10-17", 2100000)])
        for options in ({"require_fresh": True}, {"require_complete": True}):
            with self.subTest(options=options), patch("data_retrieval.retrieve_tsa_official_year", side_effect=URLError("blocked")), patch("data_retrieval.download", return_value=(json.dumps(self.payload).encode(), dict(self.entry))):
                log = []
                with self.assertRaisesRegex(ValueError, "strict checks"):
                    retrieve_tsa(2026, 2026, self.directory, 20, log, **options)
                self.assertEqual(log[-1]["status"], "rejected")

    def test_narrow_cli_refresh_preserves_other_years_and_structured_log(self):
        historical = self.saved_row("2025-12-31", 2200000)
        current = self.saved_row("2026-11-03", 1966283)
        write_csv(self.directory / "tsa_daily.csv", [historical, current])
        args = ["data_retrieval.py", "--sources", "tsa", "--start-year", "2026", "--output-dir", str(self.directory)]
        with patch.object(sys, "argv", args), patch("data_retrieval.retrieve_tsa_official_year", side_effect=URLError("blocked")), patch("data_retrieval.download", side_effect=OSError("Backup unavailable")):
            self.assertEqual(retrieval_main(), 0)
        rows = pd.read_csv(self.directory / "tsa_daily.csv", dtype=str, keep_default_na=False)
        self.assertEqual(rows["date"].tolist(), ["2025-12-31", "2026-11-03"])
        log = pd.read_csv(self.directory / "retrieval_log.csv", dtype=str, keep_default_na=False)
        self.assertEqual(json.loads(log.iloc[-1]["details_json"])["recovery"], "cache_only")


class HistoricalEvaluationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        fixture = BaselineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.base = fixture.input_path.parent
        (self.base / "input_data").mkdir()
        self.financial_path = self.base / "input_data" / "delta_quarterly.csv"
        shutil.copy2(fixture.input_path, self.financial_path)
        self.output = self.base / "output_data" / "vintage_backtest"
        self.output.mkdir(parents=True)
        self.audit_path = self.output / "driver_run.json"
        self.origins = ["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01"]
        self.audit = {"cutoff_date": "2026-11-06", "signal_data_basis": "tsa_pre_cutoff_captures_and_alfred_as_of_vintages",
                      "input_csv_sha256": {"delta_quarterly.csv": hashlib.sha256(self.financial_path.read_bytes()).hexdigest()},
                      "forecasting_code_sha256": hashlib.sha256((Path(__file__).parent / "forecasting.py").read_bytes()).hexdigest(),
                      "strict_point_in_time": True, "historical_signal_evidence_verified": True,
                      "financial_source_documents_verified": True, "current_vintage_fallback_used": False,
                      "backtest_origins": [{"cutoff_date": origin, "signal_audit": {"signals_verified": True},
                                            "financial_audit": {"source_documents_verified": True,
                                                                "filing_dates_before_cutoff": True}} for origin in self.origins]}
        atomic_json(self.audit_path, self.audit)
        annotate(self.financial_path, self.audit_path)
        metrics = []
        for filename, model in (("baseline_backtest.csv", "seasonal_naive"), ("driver_backtest.csv", "tsa_airfare_ridge")):
            scored = [{"cutoff_date": origin, "horizon": horizon,
                       "quarter": str(pd.Period(origin, freq="Q-DEC") + horizon - 1),
                       "forecast_usd_millions": 100.0, "actual_usd_millions": 100.0,
                       "absolute_percentage_error": 0.0} for origin in self.origins for horizon in range(1, 7)]
            pd.DataFrame(scored).to_csv(self.output / filename, index=False)
            metrics.extend({"model": model, "horizon": horizon, "origins": 4,
                            "mape_percent": 0.0, "mae_usd_millions": 0.0} for horizon in range(1, 7))
        pd.DataFrame(metrics).to_csv(self.output / "model_comparison.csv", index=False)

    def load(self, cutoff="2026-11-07"):
        fingerprints = {"delta_quarterly.csv": hashlib.sha256(self.financial_path.read_bytes()).hexdigest()}
        return load_evaluation_snapshot(self.base, cutoff, fingerprints)

    def test_evaluation_remains_compatible_on_later_days_without_changing_its_date(self):
        for cutoff, age in (("2026-11-06", 0), ("2026-11-07", 1), ("2027-11-06", 365)):
            with self.subTest(cutoff=cutoff):
                result = self.load(cutoff)
                self.assertTrue(result["matches_snapshot_inputs"])
                self.assertTrue(result["historical_input_evidence_verified"])
                self.assertEqual(result["cutoff_date"], "2026-11-06")
                self.assertEqual(result["evaluation_age_days"], age)
                self.assertEqual(result["compatibility_issues"], [])

    def test_future_evaluation_is_incompatible(self):
        result = self.load("2026-11-05")
        self.assertFalse(result["matches_snapshot_inputs"])
        self.assertIn("dated after", " ".join(result["compatibility_issues"]))

    def test_newer_forecast_does_not_hide_changed_financial_values(self):
        rows = pd.read_csv(self.financial_path)
        rows.loc[0, REVENUE] += 1
        rows.to_csv(self.financial_path, index=False)
        result = self.load()
        self.assertFalse(result["matches_snapshot_inputs"])
        self.assertFalse(result["matches_financial_inputs"])
        self.assertIn("Financial model values", " ".join(result["compatibility_issues"]))

    def test_newer_forecast_does_not_hide_changed_numerical_code(self):
        audit = json.loads(self.audit_path.read_text())
        audit["forecasting_code_sha256"] = "f" * 64
        atomic_json(self.audit_path, audit)
        result = self.load()
        self.assertFalse(result["matches_snapshot_inputs"])
        self.assertIn("numerical model code", " ".join(result["compatibility_issues"]))

    def test_invalid_evaluation_date_and_modified_metrics_remain_unavailable(self):
        audit = json.loads(self.audit_path.read_text())
        audit["cutoff_date"] = "not-a-date"
        atomic_json(self.audit_path, audit)
        self.assertEqual(self.load()["status"], "unavailable")
        audit["cutoff_date"] = "2026-11-06"
        atomic_json(self.audit_path, audit)
        metrics = pd.read_csv(self.output / "model_comparison.csv")
        metrics.loc[0, "mape_percent"] = 99
        metrics.to_csv(self.output / "model_comparison.csv", index=False)
        self.assertEqual(self.load()["status"], "unavailable")


class FinancialEvaluationIdentityTests(unittest.TestCase):
    def setUp(self):
        fixture = BaselineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.path = fixture.input_path
        self.rows = fixture.raw.copy()
        self.audit_path = self.path.parent / "driver_run.json"
        self.original = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.audit = {"input_csv_sha256": {"delta_quarterly.csv": self.original},
                      "unchanged_evaluation": {"mape": 5.63}}
        atomic_json(self.audit_path, self.audit)
        annotate(self.path, self.audit_path)
        self.annotated = json.loads(self.audit_path.read_text())

    def match(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return financial_evaluation_match(self.annotated, self.path, digest)

    def test_identical_model_inputs_accept_changed_document_hashes_and_row_format(self):
        self.rows["source_sha256"] = "b" * 64
        self.rows[REVENUE] = self.rows[REVENUE].map(lambda x: f"{x:.2f}")
        self.rows.iloc[::-1].to_csv(self.path, index=False, lineterminator="\r\n")
        self.assertEqual(self.match(), (True, False, "model_values_dates_and_filing_links"))
        self.assertEqual(self.annotated["input_csv_sha256"], self.audit["input_csv_sha256"])
        self.assertEqual(self.annotated["unchanged_evaluation"], self.audit["unchanged_evaluation"])

    def test_revenue_dates_and_filing_links_must_match(self):
        for field, value in ((REVENUE, 9999), ("published_date", "2019-04-22"),
                             ("source_url", "https://www.sec.gov/Archives/edgar/data/27904/different.htm")):
            with self.subTest(field=field):
                changed = self.rows.copy()
                changed.loc[0, field] = value
                changed.to_csv(self.path, index=False)
                self.assertFalse(self.match()[0])

    def test_unannotated_audit_does_not_accept_different_csv(self):
        self.assertEqual(financial_evaluation_match(self.audit, self.path, "f" * 64)[:2], (False, False))

    def test_annotation_refuses_nonoriginal_csv(self):
        self.path.write_bytes(self.path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "differs from the backtested original"):
            annotate(self.path, self.audit_path)

    def test_reference_corruption_or_wrong_original_identity_is_rejected(self):
        self.annotated["financial_model_reference"]["rows"][0][REVENUE] = 9999
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.match()
        self.annotated["financial_model_reference"] = financial_model_reference(self.path)
        self.annotated["financial_model_reference"]["original_csv_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "backtested input"):
            self.match()


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.input_path = Path(self.temporary.name) / "financials.csv"
        records = []
        for index, period in enumerate(pd.period_range("2019Q1", "2026Q3", freq="Q-DEC")):
            end = period.end_time.date()
            records.append({"quarter": str(period), "period_end": end.isoformat(),
                            "published_date": (end + timedelta(days=20)).isoformat(),
                            REVENUE: 1000 + index * 10, "source_url": "https://www.sec.gov/Archives/edgar/data/27904/example.htm",
                            "additional_source_url": "", "source_sha256": "a" * 64})
        self.raw = pd.DataFrame(records)
        self.raw.to_csv(self.input_path, index=False)
        self.financials = load_financials(self.input_path)
        self.cutoff = "2026-11-04"

    def test_same_day_filings_are_excluded(self):
        known = available_financials(self.financials, "2024-10-20")
        self.assertEqual(str(known["period"].iloc[-1]), "2024Q2")
        self.assertTrue((known["published_date"] < date(2024, 10, 20)).all())

    def test_six_quarters_repeat_last_four_and_recurse(self):
        forecast = seasonal_naive(self.financials, self.cutoff)
        values = self.financials[REVENUE].tail(4).tolist()
        self.assertEqual(forecast["forecast_usd_millions"].tolist(), values + values[:2])
        self.assertEqual(forecast["quarter"].tolist(), ["2026Q4", "2027Q1", "2027Q2", "2027Q3", "2027Q4", "2028Q1"])
        self.assertEqual(forecast["reference_type"].tolist(), ["observed"] * 4 + ["recursive_forecast"] * 2)
        self.assertTrue((forecast["input_published_date"] < self.cutoff).all())

    def test_future_actuals_cannot_change_forecasts(self):
        origin = "2024-06-01"
        expected = seasonal_naive(self.financials, origin)
        poisoned = self.financials.copy()
        poisoned.loc[poisoned["published_date"] >= date.fromisoformat(origin), REVENUE] = 999999999
        pd.testing.assert_frame_equal(expected, seasonal_naive(poisoned, origin))

    def test_default_backtest_scores_four_origins_and_six_horizons(self):
        origins = default_origins(self.financials, self.cutoff)
        self.assertEqual(origins, ["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01"])
        detail, metrics = backtest(self.financials, origins, self.cutoff)
        self.assertEqual(len(detail), 24)
        self.assertEqual(metrics["horizon"].tolist(), list(range(1, 7)))
        self.assertEqual(metrics["origins"].tolist(), [4] * 6)
        self.assertTrue((detail["input_published_date"] < detail["cutoff_date"]).all())
        self.assertTrue((detail["actual_published_date"] < self.cutoff).all())
        expected_mape = detail.loc[detail["horizon"] == 6, "absolute_percentage_error"].mean()
        self.assertAlmostEqual(metrics.loc[metrics["horizon"] == 6, "mape_percent"].iloc[0], expected_mape)

    def test_unavailable_outcomes_and_duplicate_origins_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "six outcomes"):
            backtest(self.financials, ["2026-09-01"], self.cutoff)
        with self.assertRaisesRegex(ValueError, "unique"):
            backtest(self.financials, ["2024-06-01", "2024-06-01"], self.cutoff)
        with self.assertRaisesRegex(ValueError, "precede"):
            backtest(self.financials, [self.cutoff], self.cutoff)

    def test_invalid_input_is_rejected(self):
        cases = []
        cases.append(pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True))
        cases.append(self.raw.drop(index=5))
        for value in ("NaN", "inf", "-1", "0"):
            invalid = self.raw.copy()
            invalid[REVENUE] = invalid[REVENUE].astype(str)
            invalid.loc[0, REVENUE] = value
            cases.append(invalid)
        for column, value in (("published_date", "2019-03-31"), ("period_end", "2019-03-30"),
                              ("source_url", "https://example.com"), ("source_sha256", "bad")):
            invalid = self.raw.copy()
            invalid.loc[0, column] = value
            cases.append(invalid)
        for invalid in cases:
            with self.subTest(case=len(invalid), first=invalid.iloc[0].to_dict()):
                invalid.to_csv(self.input_path, index=False)
                with self.assertRaises(ValueError):
                    load_financials(self.input_path)

    def test_insufficient_or_gapped_available_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four available"):
            seasonal_naive(self.financials, "2019-11-01")
        delayed = self.financials.copy()
        delayed.loc[delayed["quarter"] == "2024Q1", "published_date"] = date(2026, 12, 1)
        with self.assertRaisesRegex(ValueError, "gaps"):
            seasonal_naive(delayed, "2024-09-01")

    def test_cli_artifacts_are_explicitly_baseline_only(self):
        output = Path(self.temporary.name) / "outputs"
        run_baseline(self.input_path, output, self.cutoff)
        audit = json.loads((output / "baseline_run.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["submission_ready"])
        self.assertFalse(audit["external_signal_backtest_ready"])
        self.assertEqual(audit["external_signals_used"], [])
        self.assertEqual(len(audit["backtest_origins"]), 4)
        self.assertEqual(len(audit["input_csv_sha256"]), 64)
        self.assertEqual(len(pd.read_csv(output / "baseline_backtest.csv")), 24)

    def test_explicit_origins_do_not_call_automatic_selection(self):
        output = Path(self.temporary.name) / "explicit"
        origins = ["2024-06-01", "2024-09-01", "2024-12-01", "2025-03-01"]
        with patch("forecasting.default_origins", side_effect=AssertionError("Unexpected automatic selection")):
            run_baseline(self.input_path, output, self.cutoff, origins=origins)
        audit = json.loads((output / "baseline_run.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["origin_selection"], "Explicit user-supplied cutoffs")


class DriverTests(unittest.TestCase):
    def setUp(self):
        BaselineTests.setUp(self)
        days = pd.date_range("2019-01-01", "2026-11-30", freq="D")
        months = pd.date_range("2019-01-01", "2026-11-01", freq="MS")
        self.tsa_path = Path(self.temporary.name) / "tsa.csv"
        self.airfare_path = Path(self.temporary.name) / "airfare.csv"
        pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "passengers": [1000000 + index * 100 for index in range(len(days))],
                      "source_url": "https://example.com/tsa", "source_sha256": "a" * 64}).to_csv(self.tsa_path, index=False)
        pd.DataFrame({"date": months.strftime("%Y-%m-%d"), "airfare_cpi_index": [200 + index * 0.3 + index ** 2 * 0.001 for index in range(len(months))],
                      "value_status": "reported", "source_url": "https://example.com/airfare", "source_sha256": "b" * 64}).to_csv(self.airfare_path, index=False)
        self.signals = load_signals(self.tsa_path, self.airfare_path)

    def test_quarterly_aggregation_excludes_partial_and_same_day_data(self):
        quarters = quarterly_signals(self.signals, "2024-06-01")
        self.assertTrue(pd.isna(quarters.loc[pd.Period("2024Q2"), "tsa"]))
        expected = self.signals["tsa"].loc[self.signals["tsa"]["date"].between("2024-01-01", "2024-03-31"), "value"].sum()
        self.assertAlmostEqual(quarters.loc[pd.Period("2024Q1"), "tsa"], expected)

    def test_future_financials_and_signals_cannot_change_driver_forecast(self):
        origin = "2024-09-01"
        expected, fit, training, assumptions = driver_forecast(self.financials, self.signals, origin)
        poisoned = self.financials.copy()
        poisoned.loc[poisoned["published_date"] >= date.fromisoformat(origin), REVENUE] *= 100
        signals = {name: frame.copy() for name, frame in self.signals.items()}
        for frame in signals.values():
            frame.loc[frame["date"] >= pd.Timestamp(origin), "value"] *= 100
        actual, changed_fit, changed_training, changed_assumptions = driver_forecast(poisoned, signals, origin)
        pd.testing.assert_frame_equal(expected, actual)
        pd.testing.assert_frame_equal(training, changed_training)
        self.assertEqual(fit, changed_fit)
        self.assertEqual(assumptions, changed_assumptions)
        self.assertLessEqual(fit["training_quarters"], 12)
        self.assertGreaterEqual(fit["tsa_coefficient"], 0)
        self.assertGreaterEqual(fit["airfare_coefficient"], 0)

    def test_flagged_airfare_blank_is_not_filled(self):
        signals = {name: frame.copy() for name, frame in self.signals.items()}
        signals["airfare"].loc[signals["airfare"]["date"] == pd.Timestamp("2024-01-01"), "value"] = float("nan")
        quarters = quarterly_signals(signals, self.cutoff)
        self.assertTrue(pd.isna(quarters.loc[pd.Period("2024Q1"), "airfare"]))
        fit, training, quarters = fit_driver(self.financials, signals, self.cutoff)
        self.assertNotIn(pd.Period("2024Q1"), training.index)
        self.assertNotIn(pd.Period("2025Q1"), training.index)

    def test_recursive_forecasts_and_positive_sensitivities(self):
        forecast, fit, training, assumptions = driver_forecast(self.financials, self.signals, self.cutoff)
        self.assertAlmostEqual(forecast.iloc[4]["seasonal_anchor_usd_millions"], forecast.iloc[0]["forecast_usd_millions"])
        self.assertAlmostEqual(forecast.iloc[5]["seasonal_anchor_usd_millions"], forecast.iloc[1]["forecast_usd_millions"])
        self.assertEqual(forecast["near_term_weight"].tolist(), [1, 0.8, 0.6, 0.4, 0.2, 0])
        higher, fit, training, assumptions = driver_forecast(self.financials, self.signals, self.cutoff, volume_shift_pp=1, price_shift_pp=1)
        self.assertTrue((higher["forecast_usd_millions"] >= forecast["forecast_usd_millions"] - 1e-8).all())
        self.assertTrue(forecast["revenue_growth_percent"].between(-30.000001, 30.000001).all())

    def test_older_tsa_requires_explicit_structural_mode(self):
        signals = {name: frame.copy() for name, frame in self.signals.items()}
        signals["tsa"] = signals["tsa"].loc[signals["tsa"]["date"] < pd.Timestamp("2026-10-20")]
        with self.assertRaisesRegex(ValueError, "stale"):
            driver_forecast(self.financials, signals, self.cutoff)
        with self.assertRaisesRegex(ValueError, "longer-run"):
            driver_forecast(self.financials, signals, self.cutoff, allow_stale_tsa=True)
        forecast, fit, training, assumptions = driver_forecast(
            self.financials, signals, self.cutoff, signal_actions={"tsa": "use_structural", "airfare": "use_recent"},
            allow_stale_tsa=True)
        self.assertTrue(assumptions["tsa"]["stale"])
        self.assertEqual(assumptions["tsa"]["observation_age_days"], 16)
        self.assertTrue(forecast["tsa_action"].eq("use_structural").all())
        self.assertTrue(forecast["tsa_growth_percent"].round(9).eq(forecast.iloc[0]["tsa_growth_percent"].round(9)).all())
        signals["tsa"] = signals["tsa"].loc[signals["tsa"]["date"] != pd.Timestamp("2026-10-18")]
        with self.assertRaisesRegex(ValueError, "28 TSA days"):
            driver_forecast(self.financials, signals, self.cutoff,
                            signal_actions={"tsa": "use_structural", "airfare": "use_recent"}, allow_stale_tsa=True)

    def test_signal_schema_and_short_history_are_rejected(self):
        raw = pd.read_csv(self.airfare_path)
        raw.loc[0, "airfare_cpi_index"] = -1
        raw.to_csv(self.airfare_path, index=False)
        with self.assertRaisesRegex(ValueError, "Invalid airfare"):
            load_signals(self.tsa_path, self.airfare_path)
        with self.assertRaisesRegex(ValueError, "eight complete"):
            driver_forecast(self.financials, self.signals, "2023-06-01")

    def test_growth_guardrails_are_recorded(self):
        fit, training, quarters = fit_driver(self.financials, self.signals, self.cutoff)
        fit["intercept_log_pp"] = 1000
        with patch("forecasting.fit_driver", return_value=(fit, training, quarters)):
            forecast, fit, training, assumptions = driver_forecast(self.financials, self.signals, self.cutoff)
        self.assertTrue(forecast.iloc[0]["growth_guardrail_hit"])
        self.assertAlmostEqual(forecast.iloc[0]["revenue_growth_percent"], 30)
        self.assertGreater(forecast.iloc[0]["raw_revenue_growth_percent"], 30)
        with self.assertRaisesRegex(ValueError, "greater than -100%"):
            driver_forecast(self.financials, self.signals, self.cutoff, volume_shift_pp=-200)

    def test_bounded_signal_decisions_change_assumptions_not_coefficients(self):
        original, original_fit, training, assumptions = driver_forecast(self.financials, self.signals, self.cutoff)
        explicit, fit, training, assumptions = driver_forecast(self.financials, self.signals, self.cutoff,
                                                              signal_actions={"tsa": "use_recent", "airfare": "use_recent"})
        pd.testing.assert_frame_equal(original, explicit)
        changed, fit, training, assumptions = driver_forecast(self.financials, self.signals, self.cutoff,
                                                             signal_actions={"tsa": "use_structural", "airfare": "use_recent"})
        self.assertEqual(original_fit, fit)
        self.assertNotEqual(original.iloc[0]["forecast_usd_millions"], changed.iloc[0]["forecast_usd_millions"])
        self.assertTrue(changed["tsa_growth_percent"].round(9).eq(changed.iloc[0]["tsa_growth_percent"].round(9)).all())
        with self.assertRaisesRegex(ValueError, "Signal actions"):
            driver_forecast(self.financials, self.signals, self.cutoff, signal_actions={"tsa": "invent", "airfare": "use_recent"})

    def test_ranges_are_positive_widening_and_cutoff_scoped(self):
        forecast, fit, training, assumptions = driver_forecast(self.financials, self.signals, "2024-09-01")
        ranged = add_driver_ranges(forecast, self.financials, "2024-09-01", fit)
        self.assertTrue((ranged["lower_80_usd_millions"] >= 0).all())
        self.assertTrue((ranged["lower_80_usd_millions"] < ranged["forecast_usd_millions"]).all())
        self.assertTrue((ranged["upper_80_usd_millions"] > ranged["forecast_usd_millions"]).all())
        self.assertTrue((ranged["interval_half_width_usd_millions"].diff().dropna() > 0).all())
        poisoned = self.financials.copy()
        poisoned.loc[poisoned["published_date"] >= date(2024, 9, 1), REVENUE] *= 100
        pd.testing.assert_frame_equal(ranged, add_driver_ranges(forecast, poisoned, "2024-09-01", fit))

    def test_common_origins_and_model_comparison(self):
        origins = default_origins(self.financials, self.cutoff)
        detail, metrics, fits = backtest_driver(self.financials, self.signals, origins, self.cutoff)
        baseline_detail, baseline_metrics = backtest(self.financials, origins, self.cutoff)
        self.assertEqual(len(detail), 24)
        self.assertEqual(len(metrics), 12)
        self.assertEqual(len(fits), 4)
        self.assertEqual(detail[["cutoff_date", "quarter", "horizon"]].to_dict("records"),
                         baseline_detail[["cutoff_date", "quarter", "horizon"]].to_dict("records"))
        self.assertTrue(detail["actual_published_date"].lt(self.cutoff).all())
        self.assertTrue(metrics["origins"].eq(4).all())
        self.assertTrue(metrics.loc[metrics["model"] == "tsa_airfare_ridge", "empirical_range_coverage"].between(0, 1).all())

    def test_driver_outputs_and_sensitivities_are_explicitly_provisional(self):
        output = Path(self.temporary.name) / "drivers"
        forecast, detail, comparison = run_driver_model(self.input_path, self.tsa_path, self.airfare_path, output, self.cutoff)
        audit = json.loads((output / "driver_run.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["strict_point_in_time"])
        self.assertFalse(audit["submission_ready"])
        self.assertEqual(audit["external_signals_used"], ["tsa", "airfare"])
        self.assertEqual(audit["approval_status"], "not_reviewed")
        scenarios = pd.read_csv(output / "driver_sensitivities.csv")
        self.assertEqual(len(scenarios), 30)
        self.assertTrue(scenarios.loc[scenarios["scenario"] == "base", "change_usd_millions"].eq(0).all())
        self.assertTrue((scenarios.loc[scenarios["scenario"].isin(["volume_plus_1pp", "pricing_plus_1pp"]), "change_usd_millions"] >= -1e-8).all())
        self.assertEqual(len(pd.read_csv(output / "driver_training.csv")), audit["fit"]["training_quarters"])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        DriverTests.setUp(self)
        self.base_dir = Path(self.temporary.name) / "project"
        data_dir = self.base_dir / "input_data"
        data_dir.mkdir(parents=True)
        for source, name in ((self.input_path, "delta_quarterly.csv"), (self.tsa_path, "tsa_daily.csv"),
                             (self.airfare_path, "airfare_cpi_monthly.csv")):
            shutil.copyfile(source, data_dir / name)
        self.context = prepare_context(self.base_dir, self.cutoff)
        self.agent = {"model": "synthetic-test", "answer": {
            "summary": "Use long-run pricing while retaining recent volume.", "cautions": ["Historical vintages are unresolved."],
            "decisions": {"tsa": {"action": "use_recent", "reason": "Recent volume is available.", "evidence_ids": ["TSA_RECENT"]},
                          "airfare": {"action": "use_structural", "reason": "Use a conservative pricing assumption.", "evidence_ids": ["AIRFARE_STRUCTURAL"]}}}}
        self.directory = self.base_dir / "output_data" / "review_runs"

    def test_agent_decisions_require_known_signal_evidence_and_actions(self):
        validate_decisions(self.agent["answer"])
        for field, value in (("action", "publish"), ("evidence_ids", ["INVENTED"]), ("evidence_ids", ["TSA_RECENT"])):
            answer = json.loads(json.dumps(self.agent["answer"]))
            answer["decisions"]["airfare"][field] = value
            with self.assertRaises(ValueError):
                validate_decisions(answer)

    def test_draft_does_not_publish_and_overrides_need_a_reason(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        save_draft(self.directory, draft)
        self.assertFalse((self.directory / "approved.json").exists())
        self.assertEqual(load_record(self.directory / "draft.json")["run_id"], draft["run_id"])
        with self.assertRaisesRegex(ValueError, "reason"):
            build_draft(self.base_dir, self.context, self.agent, volume_shift_pp=1)
        with self.assertRaisesRegex(ValueError, "-5 to"):
            build_draft(self.base_dir, self.context, self.agent, volume_shift_pp=6, override_reason="Stress")
        updated = build_draft(self.base_dir, self.context, self.agent, previous=draft,
                              volume_shift_pp=1, override_reason="Higher relative volume growth")
        self.assertNotEqual(updated["run_id"], draft["run_id"])
        self.assertEqual(len(updated["changes"]["quarter_changes"]), 6)
        self.assertGreater(updated["forecast"][-1]["forecast_usd_millions"], draft["forecast"][-1]["forecast_usd_millions"])

    def test_isolated_runs_do_not_replace_working_draft_and_history_is_frozen(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        save_draft(self.directory, draft)
        isolated = build_draft(self.base_dir, self.context, self.agent, previous=draft)
        save_draft(self.directory, isolated, publish_working=False)
        self.assertEqual(load_record(self.directory / "draft.json")["run_id"], draft["run_id"])
        self.assertTrue((self.directory / "runs" / f"{isolated['run_id']}.json").exists())
        self.assertEqual(len(draft["context"]["reported_history"]), 8)
        original = draft["context"]["reported_history"][-1][REVENUE]
        rows = pd.read_csv(self.base_dir / "input_data" / "delta_quarterly.csv")
        rows.loc[len(rows) - 1, REVENUE] *= 2
        rows.to_csv(self.base_dir / "input_data" / "delta_quarterly.csv", index=False)
        self.assertEqual(load_record(self.directory / "draft.json")["context"]["reported_history"][-1][REVENUE], original)

    def test_narrative_is_post_calculation_bounded_and_failure_is_explicit(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        request = narrative_context(draft)
        self.assertNotIn("forecast", request)
        self.assertTrue(request["verified_observations"])
        self.assertEqual(request["comparison_status"], "no_previous")
        self.assertNotIn("reason", request["growth_adjustments_pp"])
        result = {"model": "synthetic", "answer": {"observed_changes": ["First recorded forecast."],
                  "interpretation": "Long-run pricing was applied.", "cautions": ["Illustrative ranges."]}}
        explained = attach_narrative(draft, result)
        self.assertEqual(explained["change_narrative"]["status"], "complete")
        self.assertEqual(explained["forecast"], draft["forecast"])
        self.assertEqual(len(draft["sensitivities"]), 4)
        failed = attach_narrative(draft, failure=True)
        self.assertEqual(failed["change_narrative"]["status"], "failed")
        self.assertEqual(failed["change_narrative"]["interpretation"], "")

    def test_change_amounts_are_formatted_by_python_and_ai_amounts_are_ignored(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        draft["changes"]["previous_run_id"] = "previous-test-run"
        first, last = draft["forecast"][0]["quarter"], draft["forecast"][-1]["quarter"]
        draft["changes"]["quarter_changes"] = [
            {"quarter": first, "change_usd_millions": -4.574732453060278},
            {"quarter": last, "change_usd_millions": -3.557666477809107}]
        explained = attach_narrative(draft, {"answer": {
            "observed_changes": ["Revenue declined by 4.57 billion."],
            "interpretation": "Recent demand informs the forecast while longer-run pricing moderates unusual fare movements.",
            "cautions": ["All forecasts are certain."]}})
        text = " ".join(explained["change_narrative"]["observed_changes"])
        self.assertIn("$4.57m lower", text)
        self.assertIn("$3.56m lower", text)
        self.assertNotIn("4.57 billion", text)
        self.assertNotIn("All forecasts are certain.", explained["change_narrative"]["cautions"])
        self.assertEqual(explained["forecast"], draft["forecast"])

    def test_ai_interpretation_cannot_reintroduce_currency_or_clipping_claims(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        for text in ("Forecast fell by $4.57 billion.", "Seed clipping differs from assumption clipping."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                attach_narrative(draft, {"answer": {"interpretation": text}})

    def test_latest_saved_run_is_selected_after_restart_and_not_by_approval_time(self):
        old = build_draft(self.base_dir, self.context, self.agent)
        old["created_at_utc"] = "2026-11-04T08:00:00+00:00"
        old = seal_record(old)
        newer = build_draft(self.base_dir, self.context, self.agent)
        newer["created_at_utc"] = "2026-11-04T09:00:00+00:00"
        newer = seal_record(newer)
        approved = json.loads(json.dumps(old))
        approved.update(status="approved", approved_at_utc="2026-11-04T10:00:00+00:00")
        approved = seal_record(approved)
        atomic_json(self.directory / "draft.json", old)
        atomic_json(self.directory / "latest_draft.json", newer)
        atomic_json(self.directory / "approved.json", approved)
        self.assertEqual(latest_saved_forecast(self.directory)["run_id"], newer["run_id"])
        updated = build_draft(self.base_dir, self.context, self.agent, previous=latest_saved_forecast(self.directory),
                              comparison_basis="latest_saved_run")
        self.assertEqual(updated["changes"]["previous_run_id"], newer["run_id"])
        self.assertEqual(updated["changes"]["previous_created_at_utc"], newer["created_at_utc"])
        self.assertEqual(updated["changes"]["previous_status"], "draft")
        self.assertEqual(updated["changes"]["comparison_basis"], "latest_saved_run")

    def test_public_approval_is_available_as_initial_comparison(self):
        published = self.base_dir / "output_data" / "published" / "approved.json"
        self.assertIsNone(latest_saved_forecast(self.directory, published))
        draft = build_draft(self.base_dir, self.context, self.agent)
        draft["status"] = "approved"
        atomic_json(published, seal_record(draft))
        self.assertEqual(latest_saved_forecast(self.directory, published)["run_id"], draft["run_id"])

    def test_invalid_history_cannot_silently_become_a_first_run(self):
        self.directory.mkdir(parents=True)
        (self.directory / "latest_draft.json").write_text("invalid", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "history could not be validated"):
            latest_saved_forecast(self.directory)

    def test_attempt_allowance_counts_retries_and_survives_reopening(self):
        path = self.base_dir / "calls.sqlite"
        self.assertEqual(attempt_usage(path), 0)
        self.assertEqual(consume_attempt(path, 2), 1)
        self.assertEqual(consume_attempt(path, 2), 2)
        self.assertEqual(attempt_usage(path), 2)
        with self.assertRaisesRegex(ValueError, "allowance"):
            consume_attempt(path, 2)

    def test_approval_is_separate_and_tied_to_reviewed_content(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        save_draft(self.directory, draft)
        with self.assertRaisesRegex(ValueError, "Acknowledge"):
            approve_draft(self.directory, draft, self.base_dir, "Reviewer", "Reviewed")
        approved = approve_draft(self.directory, draft, self.base_dir, "Reviewer", "Reviewed assumptions", True)
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["reviewed_record_sha256"], draft["integrity_sha256"])
        self.assertEqual(approved["forecast"], draft["forecast"])
        self.assertEqual(load_record(self.directory / "draft.json")["status"], "draft")
        changed = build_draft(self.base_dir, self.context, self.agent, previous=draft,
                              price_shift_pp=1, override_reason="Pricing scenario")
        save_draft(self.directory, changed)
        self.assertEqual(load_record(self.directory / "approved.json")["run_id"], approved["run_id"])
        with self.assertRaisesRegex(ValueError, "another session"):
            approve_draft(self.directory, draft, self.base_dir, "Reviewer", "Old review", True)

    def test_stale_tsa_requires_review_for_drafting_and_approval(self):
        source = self.base_dir / "input_data" / "tsa_daily.csv"
        rows = pd.read_csv(source)
        rows.loc[rows["date"] < "2026-10-20"].to_csv(source, index=False)
        context = prepare_context(self.base_dir, self.cutoff)
        self.assertTrue(context["tsa_freshness"]["review_required"])
        self.assertEqual(context["tsa_freshness"]["age_days_at_cutoff"], 16)
        self.assertTrue(all(row["tsa_action"] == "use_structural" for row in context["model_forecast"]))
        with self.assertRaisesRegex(ValueError, "reviewer must confirm"):
            build_draft(self.base_dir, context, self.agent)
        review = {"acknowledged": True, "reviewer": "Test reviewer", "reason": "Proceed conservatively while the source is unavailable",
                  "latest_observation": "2026-10-19", "cutoff_date": self.cutoff}
        draft = build_draft(self.base_dir, context, self.agent, tsa_review=review)
        self.assertEqual(draft["applied_actions"]["tsa"], "use_structural")
        self.assertEqual(draft["agent"]["answer"]["decisions"]["tsa"]["action"], "use_recent")
        self.assertEqual(draft["tsa_stale_review"]["reviewer"], "Test reviewer")
        with self.assertRaisesRegex(ValueError, "cannot select recent"):
            build_draft(self.base_dir, context, self.agent, tsa_review=review,
                        signal_actions={"tsa": "use_recent", "airfare": "use_recent"}, override_reason="Try recent")
        with self.assertRaisesRegex(ValueError, "confirm again"):
            build_draft(self.base_dir, context, self.agent, tsa_review={**review, "latest_observation": "2026-10-18"})
        with self.assertRaisesRegex(ValueError, "reviewer label"):
            build_draft(self.base_dir, context, self.agent, tsa_review={**review, "reason": ""})
        save_draft(self.directory, draft)
        with self.assertRaisesRegex(ValueError, "Confirm the older"):
            approve_draft(self.directory, draft, self.base_dir, "Approver", "Reviewed", True)
        approved = approve_draft(self.directory, draft, self.base_dir, "Approver", "Reviewed", True, True)
        self.assertEqual(approved["forecast"], draft["forecast"])

    def test_tsa_blocking_claim_requires_recorded_403(self):
        path = self.base_dir / "input_data" / "retrieval_log.csv"
        self.assertFalse(tsa_freshness(self.base_dir, self.cutoff)["last_request_blocked"])
        pd.DataFrame([{"dataset": "tsa_2019", "http_status": "403", "requested_url": "https://www.tsa.gov/travel/passenger-volumes/2019"}]).to_csv(path, index=False)
        self.assertTrue(tsa_freshness(self.base_dir, self.cutoff)["last_request_blocked"])
        pd.DataFrame([{"dataset": "fred_airfare_cpi", "http_status": "403", "requested_url": "https://fred.stlouisfed.org/"}]).to_csv(path, index=False)
        self.assertFalse(tsa_freshness(self.base_dir, self.cutoff)["last_request_blocked"])

    def test_approved_snapshot_does_not_expire_as_time_passes(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        save_draft(self.directory, draft)
        approved = approve_draft(self.directory, draft, self.base_dir, "Reviewer", "Reviewed snapshot", True)

        class LaterDate(date):
            @classmethod
            def today(cls):
                return cls(2026, 11, 20)

        with patch("forecast_workflow.date", LaterDate):
            loaded = load_record(self.directory / "approved.json")
        self.assertEqual(loaded["forecast"], approved["forecast"])
        self.assertEqual(loaded["cutoff_date"], self.cutoff)
        self.assertEqual(loaded["context"]["tsa_freshness"]["latest_observation"], "2026-11-03")

    def test_source_changes_and_tampered_records_block_approval(self):
        draft = build_draft(self.base_dir, self.context, self.agent)
        save_draft(self.directory, draft)
        path = self.directory / "draft.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["forecast"][0]["forecast_usd_millions"] += 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_record(path)
        source = self.base_dir / "input_data" / "tsa_daily.csv"
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            require_current(draft, self.base_dir)


if __name__ == "__main__":
    unittest.main()

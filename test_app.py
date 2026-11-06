"""Public UI, permission, privacy and bounded-run tests; all providers are mocked."""

import json
import os
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from forecast_views import chart_rows, deployment_snapshot, public_record
from forecast_workflow import (approve_draft, atomic_json, attach_narrative, build_draft, load_record,
                               prepare_context, save_draft)
from run_controls import attempt_usage, consume_attempt
import test_forecasting as fixtures
from verify_quarters import latest_release, verify

ROOT = Path(__file__).parent


class PublicAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.base_dir
        self.directory = self.fixture.directory
        self.environment = patch.dict(os.environ, {"FORECAST_ARTIFACT_DIR": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def app(self, credentials=False):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45)
        app.secrets.update({"APP_PASSWORD": "synthetic-password" if credentials else "", "GEMINI_API_KEY": "synthetic-key" if credentials else "",
                            "SEC_NAME": "Test identity", "SEC_EMAIL": "test@example.invalid", "LLM_DAILY_CALL_LIMIT": "3"})
        app.run()
        next(b for b in app.button if b.label == "Run data load & forecast").click().run()
        next(b for b in app.button if b.label == "View cached forecast").click().run()
        return app

    def approved_fixture(self):
        draft = build_draft(self.root, self.fixture.context, self.fixture.agent)
        save_draft(self.directory, draft)
        approved = approve_draft(self.directory, draft, self.root, "PRIVATE_REVIEWER", "PRIVATE_APPROVAL_NOTE", True)
        published = deployment_snapshot(approved)
        atomic_json(self.root / "output_data" / "published" / "approved.json", published)
        return draft, published

    def login_workspace(self):
        app = self.app(credentials=True)
        next(button for button in app.button if button.label == "Run data load & forecast").click().run()
        next(button for button in app.button if button.label == "Load latest data & generate forecast").click().run()
        app.text_input(key="password_input").set_value("synthetic-password")
        next(button for button in app.button if button.label == "Unlock live access").click().run()
        return app

    def test_overview_is_guidance_only_and_does_not_request_a_password(self):
        self.approved_fixture()
        with patch("requests.post") as post, patch("requests.get") as get, patch("subprocess.run") as gather:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.metric)
            self.assertFalse(app.text_input)
            self.assertEqual([b.label for b in app.button], ["Run data load & forecast"])
            self.assertTrue(any("AI assesses" in item.value for item in app.markdown))
            post.assert_not_called()
            get.assert_not_called()
            gather.assert_not_called()

    def test_methodology_describes_ai_decisions_and_human_overrides(self):
        self.approved_fixture()
        app = self.app()
        next(b for b in app.button if b.label == "Methodology").click().run()
        self.assertFalse(app.exception)
        text = " ".join(m.value for m in app.markdown)
        self.assertIn("What can AI choose?", text)
        self.assertIn("What remains controlled by Python?", text)
        self.assertIn("How can a person change it?", text)
        self.assertIn("How is publication controlled?", text)
        self.assertIn("does not independently browse", text)

    def test_summary_labels_ai_and_python_and_hides_legacy_fuel_context(self):
        draft = build_draft(self.root, self.fixture.context, self.fixture.agent)
        draft["context"]["fuel_context"] = {"observations": [{"date": "2026-10-01", "usd_per_gallon": 2.5, "source_url": "https://www.eia.gov/"}]}
        draft = attach_narrative(draft, {"answer": {"interpretation": "Recent demand informs the outlook while longer-run pricing limits dependence on unusual fare movements."}})
        save_draft(self.directory, draft)
        approved = approve_draft(self.directory, draft, self.root, "Reviewer", "Reviewed the evidence.", True)
        atomic_json(self.root / "output_data" / "published" / "approved.json", deployment_snapshot(approved))
        app = self.app()
        self.assertFalse(app.exception)
        text = " ".join(m.value for m in app.markdown)
        self.assertIn("Numerical summary · Python", text)
        self.assertIn("AI interpretation", text)
        self.assertIn("Model limitations · Python", text)
        self.assertFalse(any("Fuel" in e.label for e in app.expander))

    def test_public_empty_state_and_permissions_without_network_or_credentials(self):
        with patch("requests.post") as post, patch("requests.get") as get, patch("subprocess.run") as gather:
            app = self.app()
            self.assertFalse(app.exception)
            self.assertFalse(app.metric)
            self.assertTrue(any("No approved forecast" in item.value for item in app.info))
            next(button for button in app.button if button.label == "Run data load & forecast").click().run()
            self.assertFalse(app.exception)
            next(button for button in app.button if button.label == "Load latest data & generate forecast").click().run()
            self.assertFalse(any(button.label == "Start fresh run" for button in app.button))
            post.assert_not_called()
            get.assert_not_called()
            gather.assert_not_called()

    def test_public_approved_snapshot_is_frozen_private_and_restart_readable(self):
        draft, published = self.approved_fixture()
        self.assertNotIn("PRIVATE_REVIEWER", json.dumps(published))
        self.assertNotIn("PRIVATE_APPROVAL_NOTE", json.dumps(published))
        self.assertEqual(len(chart_rows(public_record(published))), 14)
        path = self.root / "output_data" / "published" / "approved.json"
        before = path.read_bytes()
        with patch("requests.post") as post, patch("requests.get") as get:
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(app.metric)
            self.assertEqual([tab.label for tab in app.tabs], ["Drivers & outlook", "Historical performance", "Latest run"])
            app.toggle[0].set_value(True).run()
            self.assertFalse(app.exception)
            restarted = self.app()
            self.assertFalse(restarted.exception)
            self.assertEqual(restarted.metric[0].value, app.metric[0].value)
            post.assert_not_called()
            get.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(load_record(path)["status"], "approved")

    def test_cached_draft_is_public_but_publication_requires_a_fresh_code(self):
        original, published = self.approved_fixture()
        draft = build_draft(self.root, self.fixture.context, self.fixture.agent, previous=original)
        save_draft(self.directory, draft, publish_working=False)
        atomic_json(self.directory / "latest_draft.json", draft)
        path = self.root / "output_data" / "published" / "approved.json"
        before = path.read_bytes()
        app = self.app(credentials=True)
        next(button for button in app.button if button.label == "Run data load & forecast").click().run()
        next(button for button in app.button if button.label == "Inspect cached draft").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["workspace_step"], "Inspect draft")
        next(b for b in app.button if b.label == "Review AI choices").click().run()
        self.assertTrue(next(b for b in app.button if b.label == "Recalculate as new draft").disabled)
        next(b for b in app.button if b.label == "Approve & publish").click().run()
        next(b for b in app.button if b.label == "Confirm publication").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("code not recognised" in x.value for x in app.error))
        self.assertEqual(path.read_bytes(), before)

    def test_live_run_counts_retry_and_narrative_and_preserves_approval(self):
        old_draft, published = self.approved_fixture()
        path = self.root / "output_data" / "published" / "approved.json"
        before = path.read_bytes()
        decision = Mock(status_code=200)
        decision.json.return_value = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(self.fixture.agent["answer"])}]}}]}
        explanation = Mock(status_code=200)
        explanation.json.return_value = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({"observed_changes": ["No earlier comparable choices changed."], "interpretation": "Structural pricing is used.", "cautions": ["Current-vintage data."]})}]}}]}
        verification = {"status": "verified", "as_of_date": date.today().isoformat(),
                        "forecast_quarters": [row["quarter"] for row in published["forecast"]]}
        app = self.login_workspace()
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), patch("verify_quarters.verify", return_value=verification), patch("requests.post", side_effect=[Mock(status_code=503), decision, explanation]) as post:
            next(button for button in app.button if button.label == "Start fresh run").click().run(timeout=60)
            self.assertFalse(app.exception, app.exception)
            self.assertEqual(post.call_count, 3)
        self.assertEqual(attempt_usage(self.directory / "llm_attempts.sqlite"), 3)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(load_record(self.directory / "draft.json")["run_id"], old_draft["run_id"])
        self.assertEqual(app.session_state["workspace_record"]["change_narrative"]["status"], "complete")
        self.assertEqual(app.session_state["workspace_step"], "Inspect draft")

    def test_limit_and_source_failure_retain_saved_results_without_llm_calls(self):
        draft, published = self.approved_fixture()
        approved_path = self.root / "output_data" / "published" / "approved.json"
        original = approved_path.read_bytes()
        for outcome in ("limited", "source_failed"):
            with self.subTest(outcome=outcome):
                st.cache_resource.clear()
                if outcome == "limited":
                    for attempt in range(3):
                        consume_attempt(self.directory / "llm_attempts.sqlite", 3)
                app = self.login_workspace()
                verification = {"status": "verified", "as_of_date": date.today().isoformat(),
                                "forecast_quarters": [row["quarter"] for row in published["forecast"]]}
                with patch("subprocess.run", return_value=SimpleNamespace(returncode=1 if outcome == "source_failed" else 0, stdout="", stderr="")), patch("verify_quarters.verify", return_value=verification), patch("requests.post") as post:
                    next(button for button in app.button if button.label == "Start fresh run").click().run()
                    self.assertFalse(app.exception)
                    self.assertTrue(app.error)
                    post.assert_not_called()
                self.assertEqual(approved_path.read_bytes(), original)
                self.assertEqual(load_record(self.directory / "draft.json")["run_id"], draft["run_id"])

    def test_new_stale_draft_pauses_before_ai_without_reviewer_acknowledgement(self):
        import pandas as pd

        source = self.root / "input_data" / "tsa_daily.csv"
        frame = pd.read_csv(source)
        frame.loc[frame["date"] < "2026-10-20"].to_csv(source, index=False)
        app = self.login_workspace()
        self.assertTrue(any("TSA data needs review" in item.value for item in app.warning))
        verification = {"status": "verified", "as_of_date": date.today().isoformat(),
                        "forecast_quarters": [row["quarter"] for row in self.fixture.context["model_forecast"]]}
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), patch("verify_quarters.verify", return_value=verification), patch("requests.post") as post:
            next(button for button in app.button if button.label == "Start fresh run").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("reviewer must confirm" in item.value for item in app.error))
            post.assert_not_called()

    def test_reviewer_recalculation_creates_draft_then_explicitly_publishes(self):
        draft, published = self.approved_fixture()
        published_path = self.root / "output_data" / "published" / "approved.json"
        before = published_path.read_bytes()
        app = self.login_workspace()
        app.session_state["workspace_record"] = draft
        app.session_state["workspace_step"] = "Review & approve"
        app.run()
        self.assertFalse(app.exception)
        app.number_input[0].set_value(1.0)
        next(item for item in app.text_area if item.label == "Why are you changing these assumptions?").set_value("PRIVATE_OVERRIDE_NOTE")
        response = Mock(status_code=200)
        response.json.return_value = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({"observed_changes": ["Demand assumption increased."], "interpretation": "The scenario is higher, without exact causal attribution.", "cautions": ["Illustrative range."]})}]}}]}
        with patch("requests.post", return_value=response):
            next(button for button in app.button if button.label == "Recalculate as new draft").click().run()
        self.assertFalse(app.exception, app.exception)
        revised = app.session_state["workspace_record"]
        self.assertNotEqual(revised["run_id"], draft["run_id"])
        self.assertEqual(published_path.read_bytes(), before)
        next(button for button in app.button if button.label == "Approve & publish").click().run()
        next(item for item in app.text_input if item.label == "Reviewer label (name or initials)").set_value("PRIVATE_APPROVER")
        next(item for item in app.text_area if item.label == "Approval note").set_value("PRIVATE_APPROVAL_NOTE")
        next(item for item in app.text_input if item.label == "Re-enter access code to publish").set_value("synthetic-password")
        next(item for item in app.checkbox if item.label.startswith("I reviewed this saved draft")).check()
        next(button for button in app.button if button.label == "Confirm publication").click().run()
        self.assertFalse(app.exception, app.exception)
        approved = load_record(published_path)
        self.assertEqual(approved["forecast"], revised["forecast"])
        text = published_path.read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_APPROVER", text)
        self.assertNotIn("PRIVATE_APPROVAL_NOTE", text)
        self.assertNotIn("PRIVATE_OVERRIDE_NOTE", text)

    def test_unlocked_session_still_requires_a_fresh_code_to_publish(self):
        old, published = self.approved_fixture()
        draft = build_draft(self.root, self.fixture.context, self.fixture.agent, previous=old)
        save_draft(self.directory, draft, publish_working=False)
        path = self.root / "output_data" / "published" / "approved.json"
        before = path.read_bytes()
        app = self.login_workspace()
        app.session_state["workspace_record"] = draft
        app.session_state["workspace_step"] = "Review & approve"
        app.run()
        next(b for b in app.button if b.label == "Approve & publish").click().run()
        next(i for i in app.text_input if i.label == "Reviewer label (name or initials)").set_value("Test reviewer")
        next(i for i in app.text_area if i.label == "Approval note").set_value("Reviewed saved assumptions.")
        next(i for i in app.checkbox if i.label.startswith("I reviewed this saved draft")).check()
        next(b for b in app.button if b.label == "Confirm publication").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("code not recognised" in i.value for i in app.error))
        self.assertEqual(path.read_bytes(), before)

    def test_no_change_recalculation_does_not_consume_an_ai_attempt(self):
        draft, _ = self.approved_fixture()
        app = self.login_workspace()
        app.session_state["workspace_record"] = draft
        app.session_state["workspace_step"] = "Review & approve"
        app.run()
        with patch("requests.post") as post:
            next(b for b in app.button if b.label == "Recalculate as new draft").click().run()
        self.assertFalse(app.exception)
        post.assert_not_called()
        self.assertEqual(app.session_state["workspace_record"]["run_id"], draft["run_id"])

    def test_concurrent_working_draft_change_blocks_publication_without_overwriting_it(self):
        old, _ = self.approved_fixture()
        draft = build_draft(self.root, self.fixture.context, self.fixture.agent, previous=old)
        app = self.login_workspace()
        app.session_state["workspace_record"] = draft
        app.session_state["workspace_step"] = "Review & approve"
        app.run()
        next(b for b in app.button if b.label == "Approve & publish").click().run()
        other = build_draft(self.root, self.fixture.context, self.fixture.agent)
        atomic_json(self.directory / "draft.json", other)
        path = self.root / "output_data" / "published" / "approved.json"
        before = path.read_bytes()
        next(i for i in app.text_input if i.label == "Reviewer label (name or initials)").set_value("Test reviewer")
        next(i for i in app.text_area if i.label == "Approval note").set_value("Reviewed saved assumptions.")
        next(i for i in app.text_input if i.label == "Re-enter access code to publish").set_value("synthetic-password")
        next(i for i in app.checkbox if i.label.startswith("I reviewed this saved draft")).check()
        next(b for b in app.button if b.label == "Confirm publication").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Another review changed" in i.value for i in app.error))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(load_record(self.directory / "draft.json")["run_id"], other["run_id"])

    def test_publication_uses_saved_draft_not_unrecalculated_form_edits(self):
        old, _ = self.approved_fixture()
        draft = build_draft(self.root, self.fixture.context, self.fixture.agent, previous=old)
        app = self.login_workspace()
        app.session_state["workspace_record"] = draft
        app.session_state["workspace_step"] = "Review & approve"
        app.run()
        app.number_input[0].set_value(1.0)
        # The user opens publication without submitting the recalculation form.
        with patch("requests.post") as post:
            next(b for b in app.button if b.label == "Approve & publish").click().run()
            next(i for i in app.text_input if i.label == "Reviewer label (name or initials)").set_value("Test reviewer")
            next(i for i in app.text_area if i.label == "Approval note").set_value("Reviewed the saved draft shown in the confirmation.")
            next(i for i in app.text_input if i.label == "Re-enter access code to publish").set_value("synthetic-password")
            next(i for i in app.checkbox if i.label.startswith("I reviewed this saved draft")).check()
            next(b for b in app.button if b.label == "Confirm publication").click().run()
        self.assertFalse(app.exception)
        post.assert_not_called()
        approved = load_record(self.root / "output_data" / "published" / "approved.json")
        self.assertEqual(approved["forecast"], draft["forecast"])
        self.assertEqual(approved["overrides"]["volume_shift_pp"], 0)


class QuarterVerificationTests(unittest.TestCase):
    def test_latest_release_is_discovered_and_future_quarters_excluded(self):
        urls = ["https://news.delta.com/delta-air-lines-announces-june-quarter-2026-financial-results",
                "https://news.delta.com/delta-air-lines-announces-september-quarter-2026-financial-results",
                "https://news.delta.com/delta-air-lines-announces-march-quarter-2027-financial-results"]
        quarter, url = latest_release(urls, date(2026, 11, 4))
        self.assertEqual(str(quarter), "2026Q3")


if __name__ == "__main__":
    unittest.main()

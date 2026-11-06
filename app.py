"""Signal to Forecast: public approved results and a protected review workspace."""

import hashlib
import hmac
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from forecast_workflow import (AGENT_INSTRUCTION, NARRATIVE_INSTRUCTION, approve_draft, atomic_json,
                               attach_narrative, build_draft, decision_schema, fallback_narrative, latest_saved_forecast, load_record,
                               narrative_context, narrative_schema, prepare_context, require_current,
                               save_draft, tsa_freshness, validate_decisions, validate_tsa_review)
from forecast_views import (ACTION_LABELS, drivers, forecast_chart, headline, historical_performance,
                            latest_run, methodology, public_record, deployment_snapshot)
from run_controls import attempt_usage, consume_attempt
from verify_quarters import verify as verify_reported_quarter

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("FORECAST_ARTIFACT_DIR", str(BASE_DIR))).resolve()
DATA_DIR = ARTIFACT_DIR / "input_data"
OUTPUT_DIR = ARTIFACT_DIR / "output_data"
REVIEW_DIR = OUTPUT_DIR / "review_runs"
PUBLISHED_DIR = OUTPUT_DIR / "published"
load_dotenv(BASE_DIR / ".env", override=False)


def setting(name, default=""):
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        value = None
    return str(value) if value is not None else os.environ.get(name, default)


def redact(message):
    text = str(message)
    for name in ("APP_PASSWORD", "GEMINI_API_KEY", "SEC_NAME", "SEC_EMAIL", "APIFY_KEY"):
        value = setting(name)
        if value:
            text = text.replace(value, "[redacted]")
    return re.sub(r"(?<![A-Za-z0-9:/])(?:[A-Za-z]:[\\/]|/(?:home|Users|tmp)/)[^\s\"<>]+", "[local path removed]", text)[-12000:]


def password_token(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def is_reviewer():
    password = setting("APP_PASSWORD")
    return bool(password.strip()) and hmac.compare_digest(st.session_state.get("access_token", ""), password_token(password))


def authenticate_access(candidate):
    if time.monotonic() < st.session_state.get("login_blocked_until", 0):
        return False
    password = setting("APP_PASSWORD")
    if password.strip() and hmac.compare_digest(candidate.encode("utf-8"), password.encode("utf-8")):
        st.session_state["access_token"] = password_token(password)
        st.session_state["login_failures"] = 0
        st.session_state.pop("login_error", None)
        return True
    else:
        failures = st.session_state.get("login_failures", 0) + 1
        st.session_state["login_failures"] = failures
        st.session_state["login_error"] = "Access code not recognised."
        if failures % 5 == 0:
            st.session_state["login_blocked_until"] = time.monotonic() + 60
        return False


def check_password():
    authenticate_access(st.session_state.pop("password_input", ""))


def sign_in():
    if not setting("APP_PASSWORD").strip():
        st.info("Live access has not been configured here. You can view cached results without a code.")
    elif time.monotonic() < st.session_state.get("login_blocked_until", 0):
        st.warning("Please wait one minute before trying reviewer sign-in again.")
    else:
        st.caption("Use the access code included with this submission. The API key is already configured on the server.")
        with st.form("reviewer_login", clear_on_submit=True):
            st.text_input("Access code", type="password", key="password_input")
            st.form_submit_button("Unlock live access", on_click=check_password, icon=":material/lock_open:")
        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])


@st.cache_resource
def action_guard():
    return {"lock": threading.Lock(), "last_run": 0.0}


@contextmanager
def action_slot(network=True):
    guard = action_guard()
    if not guard["lock"].acquire(blocking=False):
        raise ValueError("Another run or review is in progress. Your existing results are unchanged; try again shortly.")
    try:
        remaining = 60 - (time.monotonic() - guard["last_run"])
        if network and remaining > 0:
            raise ValueError(f"The shared run cooldown is active. Please wait {int(remaining) + 1} seconds.")
        if network:
            guard["last_run"] = time.monotonic()
        yield
    finally:
        guard["lock"].release()


def read_snapshot(filename):
    try:
        return load_record(REVIEW_DIR / filename)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        st.warning("A saved forecast record could not be validated. It has not been replaced by another result; ask a reviewer to check the stored artifacts.")
        return None


def read_approved():
    try:
        record = load_record(PUBLISHED_DIR / "approved.json")
        return record if record is not None and record["status"] == "approved" else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        st.warning("The published snapshot could not be validated. No working draft has been substituted.")
        return None


def latest_attempt():
    try:
        value = json.loads((REVIEW_DIR / "latest_attempt.json").read_text(encoding="utf-8"))
        return {key: value.get(key) for key in ("time_utc", "status", "message", "run_id")}
    except (OSError, ValueError, AttributeError):
        return None


def refresh_data():
    if not is_reviewer():
        raise ValueError("Reviewer sign-in is required for live updates until a hard public quota is verified.")
    environment = os.environ.copy()
    for name in ("APP_PASSWORD", "GEMINI_API_KEY", "APIFY_KEY"):
        environment.pop(name, None)
    environment.update(SEC_NAME=setting("SEC_NAME"), SEC_EMAIL=setting("SEC_EMAIL"))
    if not environment["SEC_NAME"].strip() or not environment["SEC_EMAIL"].strip():
        raise ValueError("SEC identification is not configured. Existing forecasts are retained; ask the owner to configure SEC_NAME and SEC_EMAIL.")
    started = datetime.now(timezone.utc)
    command = [sys.executable, str(BASE_DIR / "data_retrieval.py"), "--download-sec", "--sources", "delta", "tsa", "airfare",
               "--output-dir", str(DATA_DIR), "--timeout", "30"]
    result = subprocess.run(command, cwd=BASE_DIR, env=environment, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=600, check=False)
    details = None
    try:
        log = pd.read_csv(DATA_DIR / "retrieval_log.csv", dtype=str, keep_default_na=False)
        rows = log.loc[log["dataset"] == "tsa_result"]
        if not rows.empty and datetime.fromisoformat(rows.iloc[-1]["retrieved_at_utc"]) >= started:
            details = json.loads(rows.iloc[-1]["details_json"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    st.session_state["private_diagnostics"] = redact(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise ValueError("A source update could not be completed. Saved forecasts were retained. Review source readiness or use a cached result.")
    return details


def gemini_retry_delay(response, default=2.0):
    """Honor provider hints; do not wait past the bounded retry window."""
    delays = [default]
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") if isinstance(headers, dict) or hasattr(headers, "get") else None
    if isinstance(retry_after, str):
        try:
            delays.append(float(retry_after))
        except ValueError:
            try:
                stamp = parsedate_to_datetime(retry_after)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                delays.append((stamp - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    try:
        body = response.json()
        for detail in body.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("google.rpc.RetryInfo"):
                duration = detail.get("retryDelay", "")
                if isinstance(duration, str) and re.fullmatch(r"\d+(?:\.\d+)?s", duration):
                    delays.append(float(duration[:-1]))
            if detail.get("@type", "").endswith("google.rpc.QuotaFailure"):
                if any("perday" in str(v.get("quotaId", "")).lower().replace("_", "")
                       for v in detail.get("violations", [])):
                    return None
    except (ValueError, TypeError, AttributeError):
        pass
    delay = max(delays)
    return delay if 0 <= delay <= 45 else None


def request_gemini(context, schema, instruction, validate_answer=None):
    st.session_state["last_ai_diagnostics"] = []
    if not is_reviewer():
        raise ValueError("Reviewer sign-in is required for live AI requests.")
    api_key = setting("GEMINI_API_KEY")
    model = setting("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    if not api_key or not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ValueError("The AI connection is not configured. Existing results remain available.")
    try:
        allowance = int(setting("LLM_DAILY_CALL_LIMIT", "20"))
    except ValueError:
        raise ValueError("The shared AI-call allowance is not configured correctly.") from None
    payload = {"systemInstruction": {"parts": [{"text": instruction}]},
               "contents": [{"role": "user", "parts": [{"text": json.dumps(context, allow_nan=False)}]}],
               "generationConfig": {"temperature": 0, "maxOutputTokens": 2048,
                                    "responseMimeType": "application/json", "responseSchema": schema}}
    diagnostics = st.session_state["last_ai_diagnostics"]
    for attempt in range(1, 3):
        response = None
        retryable = False
        delay = 2.0
        failure = "The AI response was incomplete or invalid."
        entry = {"attempt": attempt}
        try:
            consume_attempt(REVIEW_DIR / "llm_attempts.sqlite", allowance)
        except ValueError:
            entry["outcome"] = "local_call_allowance_reached"
            diagnostics.append(entry)
            raise
        except sqlite3.Error:
            raise ValueError("The AI-call allowance could not be verified. No further request was made.") from None
        try:
            response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                     headers={"x-goog-api-key": api_key}, json=payload, timeout=(10, 45))
            entry["http_status"] = response.status_code
            if response.status_code != 200:
                entry["outcome"] = "http_error"
                failure = f"AI request unavailable (HTTP {response.status_code})."
                retryable = response.status_code in (408, 429, 500, 502, 503, 504)
                delay = gemini_retry_delay(response)
            else:
                retryable = True
                entry["outcome"] = "invalid_response"
                body = response.json()
                if body.get("promptFeedback", {}).get("blockReason"):
                    retryable = False
                    entry["outcome"] = "blocked_response"
                    failure = "The provider blocked the AI response."
                    raise ValueError("Blocked response")
                candidate = body["candidates"][0]
                finish = candidate.get("finishReason")
                entry["finish_reason"] = finish if isinstance(finish, str) else "missing"
                if finish != "STOP":
                    retryable = finish == "MAX_TOKENS" or finish is None
                    entry["outcome"] = "incomplete_response"
                    failure = f"The AI response did not complete (finish reason: {entry['finish_reason']})."
                    if finish == "MAX_TOKENS":
                        payload["generationConfig"]["maxOutputTokens"] = 4096
                    raise ValueError("Incomplete response")
                text = "".join(part.get("text", "") for part in candidate["content"]["parts"] if not part.get("thought"))
                answer = json.loads(text)
                if not isinstance(answer, dict):
                    raise ValueError("Invalid object")
                if validate_answer:
                    validate_answer(answer)
                entry["outcome"] = "complete"
                diagnostics.append(entry)
                return {"model": model, "resolved_model": body.get("modelVersion", ""), "answer": answer,
                        "time_utc": datetime.now(timezone.utc).isoformat(), "attempts": attempt,
                        "diagnostics": list(diagnostics), "usage": body.get("usageMetadata", {}),
                        "prompt_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()}
        except requests.exceptions.JSONDecodeError:
            retryable = True
            entry["outcome"] = "invalid_json"
            failure = "The AI response was not valid JSON."
        except requests.RequestException as error:
            retryable = isinstance(error, (requests.Timeout, requests.ConnectionError))
            entry["outcome"] = "timeout" if isinstance(error, requests.Timeout) else "connection_error"
            failure = "The AI service could not be reached within the request limit."
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            # Keep provider payloads and request contents out of saved diagnostics.
            pass
        finally:
            if response is not None:
                response.close()
        diagnostics.append(entry)
        if not retryable or attempt == 2 or delay is None:
            raise ValueError(failure + " Existing numerical results and approvals are retained.") from None
        entry["retry_wait_seconds"] = delay
        st.caption(f"The AI request needs another attempt. Retrying in {delay:g} seconds; the calculated forecast is retained.")
        time.sleep(delay)


def explain_draft(record, previous):
    st.session_state.pop("narrative_error", None)
    try:
        result = request_gemini(
            narrative_context(record, previous),
            narrative_schema(),
            NARRATIVE_INSTRUCTION,
            validate_answer=lambda answer: attach_narrative(record, {"answer": answer}),
        )
        return attach_narrative(record, result)
    except (ValueError, sqlite3.Error) as error:
        st.session_state["narrative_error"] = redact(error)
        st.info("The AI explanation is unavailable. A factual summary of the calculated forecast has been saved instead.")
        return fallback_narrative(record, diagnostics=st.session_state.get("last_ai_diagnostics", []))


def save_attempt(status, message, record=None):
    summary = {"time_utc": datetime.now(timezone.utc).isoformat(), "status": status,
               "message": message, "run_id": record["run_id"] if record else None}
    atomic_json(REVIEW_DIR / "latest_attempt.json", summary)


def select_step(step):
    st.session_state["pending_workspace_step"] = step


def show_sources():
    rows = []
    for filename, label, date_column in (("delta_quarterly.csv", "Reported passenger revenue", "period_end"),
                                        ("tsa_daily.csv", "Passenger demand · TSA", "date"),
                                        ("airfare_cpi_monthly.csv", "Airfare pricing · FRED/BLS", "date")):
        try:
            frame = pd.read_csv(DATA_DIR / filename, dtype=str, keep_default_na=False)
            rows.append({"Source": label, "Latest saved observation": frame[date_column].max(), "State": "Saved data available"})
        except (OSError, ValueError, KeyError):
            rows.append({"Source": label, "Latest saved observation": "Not available", "State": "Missing or invalid; refresh required"})
    st.dataframe(rows, hide_index=True, alt="Source readiness and latest saved observation dates")


def stale_review_controls():
    try:
        freshness = tsa_freshness(ARTIFACT_DIR, date.today())
    except (OSError, ValueError, KeyError):
        return None
    if not freshness["review_required"]:
        return None
    cause = "The last recorded automated request was blocked by TSA." if freshness["last_request_blocked"] else "Newer figures have not been retrieved."
    st.warning(f"TSA data needs review. Latest observation: {freshness['latest_observation']}. {cause}")
    st.link_button("Check official TSA figures", freshness["official_url"], icon=":material/open_in_new:")
    if not is_reviewer():
        st.info("Reviewer sign-in is required to continue with older TSA data; no AI request will be made.")
        return None
    suffix = f"{date.today()}_{freshness['latest_observation']}"
    acknowledged = st.checkbox("Use older TSA data with the longer-run volume assumption", key=f"old_ack_{suffix}")
    reviewer = st.text_input("Older-data reviewer label", max_chars=80, key=f"old_reviewer_{suffix}")
    reason = st.text_area("Reason to continue with older data", max_chars=1500, key=f"old_reason_{suffix}")
    return {"acknowledged": acknowledged, "reviewer": reviewer, "reason": reason,
            "latest_observation": freshness["latest_observation"], "cutoff_date": date.today().isoformat()}


def perform_forecast_run(review):
    if not is_reviewer():
        raise ValueError("Public live execution is in cached mode until a hard provider quota is verified.")
    try:
        limit = int(setting("LLM_DAILY_CALL_LIMIT", "20"))
    except ValueError:
        raise ValueError("The AI allowance is not configured correctly. Cached results remain available.") from None
    if limit < 1 or attempt_usage(REVIEW_DIR / "llm_attempts.sqlite") >= limit:
        raise ValueError("The shared daily AI-call allowance has been reached. Cached results remain available; no source or AI request was made.")
    with action_slot(), st.status("Running forecast", expanded=True) as progress:
        try:
            previous = latest_saved_forecast(REVIEW_DIR, PUBLISHED_DIR / "approved.json")
            st.write("1. Gathering public sources")
            recovery = refresh_data()
            if recovery and recovery.get("recovery") != "official":
                st.warning(f"Data refresh completed using {recovery['recovery'].replace('_', ' ')} recovery. Latest TSA observation: {recovery.get('latest_observation', 'unknown')}. Existing newer observations were preserved.")
            st.write("2. Validating usable inputs")
            verification = verify_reported_quarter(ARTIFACT_DIR)
            if verification["status"] == "mismatch":
                raise ValueError("The latest earnings release and SEC history disagree. Update financials before generating another forecast.")
            if verification["status"] != "verified":
                st.warning("Official quarter verification could not complete. The draft will remain explicitly unverified for submission.")
            context = prepare_context(ARTIFACT_DIR, date.today())
            validate_tsa_review(context, review)
            context["quarter_verification"] = verification
            if verification["status"] == "verified" and verification["forecast_quarters"] != [row["quarter"] for row in context["model_forecast"]]:
                raise ValueError("The model targets do not match the officially verified next six quarters. Refresh the financial history.")
            st.write("3. AI recommends demand and pricing assumptions")
            agent = request_gemini({"cutoff_date": context["cutoff_date"], "evidence": context["evidence"]}, decision_schema(), AGENT_INSTRUCTION,
                                   validate_answer=validate_decisions)
            validate_decisions(agent["answer"])
            st.write("4. Python calculates the six-quarter draft")
            draft = build_draft(ARTIFACT_DIR, context, agent, previous=previous, tsa_review=review,
                                comparison_basis="latest_saved_run")
            st.write("5. Python compares the numbers; AI interprets the outlook")
            draft = explain_draft(draft, previous)
            st.write("6. Saving an isolated, unapproved run")
            save_draft(REVIEW_DIR, draft, publish_working=False)
            atomic_json(REVIEW_DIR / "latest_draft.json", draft)
            save_attempt("Draft complete", "An unapproved forecast was saved. The approved snapshot was not changed.", draft)
            st.session_state["workspace_record"] = draft
            progress.update(label="Draft saved; approval unchanged", state="complete", expanded=False)
        except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            progress.update(label="Run paused; existing results retained", state="error")
            message = "The source update exceeded its time limit. Existing forecasts remain available." if isinstance(error, subprocess.TimeoutExpired) else redact(error)
            save_attempt("Run not completed", message)
            st.error(message)
            return
    select_step("Inspect draft")
    st.rerun()


def run_forecast(review):
    try:
        perform_forecast_run(review)
    except (ValueError, OSError, sqlite3.Error) as error:
        st.error(redact(error))


def forecast_page():
    st.caption("Explore the latest explicitly approved forecast, its assumptions and the evidence behind it.")
    if st.button("Run data load & forecast", key="forecast_to_run", icon=":material/refresh:"):
        select_step("Run forecast")
        st.switch_page(RUN_PAGE)
    approved = read_approved()
    if approved is None or approved["status"] != "approved":
        st.info("No approved forecast has been published yet. A reviewer must inspect and approve a draft before an estimate appears here.")
        attempt = latest_attempt()
        if attempt:
            st.caption(f"Latest execution: {attempt.get('time_utc')} | {attempt.get('status')}")
        return
    record = public_record(approved, redact)
    headline(record)
    forecast_chart(record, "approved")
    if st.button("Methodology", icon=":material/menu_book:"):
        methodology(record)
    outlook, historical, run = st.tabs(["Drivers & outlook", "Historical performance", "Latest run"])
    with outlook:
        drivers(record, "approved")
    with historical:
        historical_performance(record)
    with run:
        latest_run(record, latest_attempt())


def inspect_draft():
    record = st.session_state.get("workspace_record")
    if record is None:
        record = read_snapshot("latest_draft.json") or read_snapshot("draft.json")
        if record:
            st.session_state["workspace_record"] = record
    if record is None:
        st.info("There is no saved draft to inspect. Generate a run when reviewer access is available.")
        return
    if st.button("Review AI choices", icon=":material/fact_check:"):
        select_step("Review & approve")
        st.rerun()
    visible = public_record(record, redact)
    st.caption("This draft is unapproved. Generating or inspecting it does not publish it.")
    headline(visible)
    forecast_chart(visible, "draft")
    approved = read_approved()
    if st.checkbox("Compare with approved forecast", disabled=approved is None):
        if approved:
            current = pd.DataFrame(record["forecast"])[["quarter", "forecast_usd_millions"]]
            old = pd.DataFrame(approved["forecast"])[["quarter", "forecast_usd_millions"]]
            joined = current.merge(old, on="quarter", suffixes=("_draft", "_approved"))
            st.caption(f"Draft cutoff {record['cutoff_date']}; approved cutoff {approved['cutoff_date']}. Only matching quarters are compared.")
            if joined.empty:
                st.info("These runs have no matching forecast quarters.")
            else:
                joined["Change (USD bn)"] = (joined["forecast_usd_millions_draft"] - joined["forecast_usd_millions_approved"]) / 1000
                figures = pd.DataFrame({"Quarter": joined["quarter"], "Draft (USD bn)": joined["forecast_usd_millions_draft"] / 1000,
                                        "Approved (USD bn)": joined["forecast_usd_millions_approved"] / 1000,
                                        "Change (USD bn)": joined["Change (USD bn)"]})
                st.dataframe(figures.round(3), hide_index=True, alt="Matching-quarter draft versus approved comparison in USD billions")
    detail = st.segmented_control("Draft detail", ["AI choices & evidence", "Run record", "Historical comparison"], default="AI choices & evidence", key="draft_detail")
    if detail == "AI choices & evidence":
        drivers(visible, "draft")
    elif detail == "Run record":
        latest_run(visible)
    else:
        historical_performance(visible)


def review_key(record):
    return record["integrity_sha256"]


def check_working_review(record):
    expected = st.session_state.get("review_working_versions", {}).get(review_key(record), "not_selected")
    current = read_snapshot("draft.json")
    actual = current["integrity_sha256"] if current else None
    if expected == "not_selected" or actual != expected:
        raise ValueError("Another review changed the working draft. Refresh this review and inspect it again before continuing.")


def approval_export():
    approved = st.session_state.get("last_approval") or read_snapshot("approved.json")
    if approved and is_reviewer():
        st.download_button("Download approved deployment snapshot", json.dumps(deployment_snapshot(approved, redact), indent=2),
                           file_name="approved.json", mime="application/json", icon=":material/download:")
        st.caption("Redacted approved export for deployment. Persistent hosting storage is configured separately.")


def close_publication():
    st.session_state.pop("publication_pending", None)


@st.dialog("Approve & publish", width="large", on_dismiss=close_publication)
def publish_dialog(expected_sha):
    record = st.session_state.get("workspace_record")
    if not record or record.get("integrity_sha256") != expected_sha:
        st.error("The selected draft changed. Close this window and inspect the current draft.")
        return
    point = record["forecast"][-1]
    period = pd.Period(point["quarter"], freq="Q-DEC")
    st.markdown("Publishing replaces the forecast displayed on the public **Approved forecast** page with this exact reviewed draft.")
    st.metric(f"Passenger revenue · Q{period.quarter} {period.year}", f"USD {point['forecast_usd_millions'] / 1000:,.3f}bn")
    st.caption(f"Draft {record['run_id'][:12]} | Data cutoff: {record['cutoff_date']}")
    st.dataframe(pd.DataFrame([
        {"Driver": label, "Saved projection strategy": ACTION_LABELS[record["applied_actions"][name]],
         "Additional growth (pp)": record["overrides"][shift]}
        for name, label, shift in (("tsa", "Demand", "volume_shift_pp"), ("airfare", "Pricing", "price_shift_pp"))
    ]), hide_index=True)
    st.info("Only the saved draft shown here will be published. Changes typed into the adjustment form must be recalculated first.")
    if not setting("APP_PASSWORD").strip():
        st.warning("Publishing access has not been configured. Cached results remain available.")
        return
    blocked = time.monotonic() < st.session_state.get("login_blocked_until", 0)
    if blocked:
        st.warning("Please wait one minute before trying the access code again.")
    with st.form(f"publish_confirmation_{record['run_id']}", clear_on_submit=True):
        reviewer = st.text_input("Reviewer label (name or initials)", max_chars=80)
        reason = st.text_area("Approval note", max_chars=1500,
                              placeholder="For example: reviewed the signal choices, forecast path and historical limitations.")
        code = st.text_input("Re-enter access code to publish", type="password")
        acknowledged = st.checkbox("I reviewed this saved draft and its assumptions, illustrative ranges and historical-test limitations. I understand it replaces the public forecast.")
        tsa_acknowledged = False
        if record.get("context", {}).get("tsa_freshness", {}).get("review_required"):
            observed = record["context"]["tsa_freshness"]["latest_observation"]
            tsa_acknowledged = st.checkbox(f"I approve using older TSA data through {observed} with the longer-run demand assumption.")
        confirm = st.form_submit_button("Confirm publication", type="primary", disabled=blocked)
    if confirm:
        if not authenticate_access(code):
            st.error(st.session_state.get("login_error", "Access code not recognised."))
            return
        if not reviewer.strip() or not reason.strip():
            st.error("Add a reviewer label and a short approval note before publishing.")
            return
        if not acknowledged:
            st.error("Confirm that you reviewed this saved draft and understand that it replaces the public forecast.")
            return
        if record.get("context", {}).get("tsa_freshness", {}).get("review_required") and not tsa_acknowledged:
            st.error("Confirm the older TSA observation date before publishing this draft.")
            return
        try:
            with action_slot(network=False):
                require_current(record, ARTIFACT_DIR)
                check_working_review(record)
                atomic_json(REVIEW_DIR / "draft.json", record)
                approved = approve_draft(REVIEW_DIR, record, ARTIFACT_DIR, reviewer, reason, acknowledged, tsa_acknowledged)
                atomic_json(PUBLISHED_DIR / "approved.json", deployment_snapshot(approved, redact))
                st.session_state["review_working_versions"][review_key(record)] = record["integrity_sha256"]
            st.session_state["last_approval"] = approved
            st.session_state["publication_notice"] = "This reviewed draft is now approved and displayed on the public Approved forecast page."
            close_publication()
            st.rerun()
        except (ValueError, OSError, KeyError) as error:
            st.error(redact(error))


def review_and_approve():
    record = st.session_state.get("workspace_record")
    if record is None:
        record = read_snapshot("latest_draft.json") or read_snapshot("draft.json")
        if record:
            st.session_state["workspace_record"] = record
    if record is None:
        st.info("Choose a cached draft or generate a fresh draft first. You can inspect it before deciding whether to change anything.")
        return
    st.caption(f"Selected draft {record['run_id'][:12]} | Generated {record['created_at_utc']} | Data cutoff {record['cutoff_date']}")
    st.info("Reviewing a draft does not change the public forecast. Keep the current assumptions, or create a revised draft below. Publishing is a separate, confirmed action.")
    try:
        require_current(record, ARTIFACT_DIR)
    except (ValueError, OSError) as error:
        st.warning("This draft no longer matches the available source files or calculation code. Generate a fresh draft before adjusting or publishing it.")
        if is_reviewer():
            with st.expander("Technical details"):
                st.text(redact(error))
        return
    working = read_snapshot("draft.json")
    versions = st.session_state.setdefault("review_working_versions", {})
    versions.setdefault(review_key(record), working["integrity_sha256"] if working else None)
    if st.button("Refresh this review", help="Reload the working-review reference after another reviewer changes it."):
        versions[review_key(record)] = working["integrity_sha256"] if working else None
        st.rerun()
    if st.session_state.get("publication_notice"):
        st.success(st.session_state["publication_notice"])
    headline(public_record(record, redact))
    preview = st.session_state.get("review_revision")
    if preview and preview["run_id"] == record["run_id"]:
        st.success(f"Revised draft saved. Its sixth-quarter estimate changed by USD {preview['change_usd_millions']:+,.1f}m versus the draft you adjusted. The public forecast was not changed by recalculation.")
        with st.expander("See the quarterly effect of your adjustments"):
            before = pd.DataFrame(preview["before"])[["quarter", "forecast_usd_millions"]]
            after = pd.DataFrame(record["forecast"])[["quarter", "forecast_usd_millions"]]
            joined = before.merge(after, on="quarter", suffixes=("_before", "_after"))
            joined["Change (USD m)"] = joined["forecast_usd_millions_after"] - joined["forecast_usd_millions_before"]
            joined = joined.rename(columns={"quarter": "Quarter", "forecast_usd_millions_before": "Before (USD m)", "forecast_usd_millions_after": "After (USD m)"})
            st.dataframe(joined.round(2), hide_index=True)
    st.markdown("### 1. Review AI choices and what is applied")
    for column, name, title in zip(st.columns(2), ("tsa", "airfare"), ("Passenger demand", "Airfare pricing")):
        recent = record["context"]["evidence"][name.upper() + "_RECENT"]
        structural = record["context"]["evidence"][name.upper() + "_STRUCTURAL"]
        decision = record["agent"]["answer"]["decisions"][name]
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(f"Observed recent growth: {recent['raw_growth_percent']:+.1f}% | Longer-run assumption: {structural['growth_percent']:+.1f}%")
            st.write("Current choice: " + ACTION_LABELS[record["applied_actions"][name]])
            st.write("AI recommendation: " + ACTION_LABELS[decision["action"]])
            if name == "tsa" and record["context"].get("tsa_freshness", {}).get("review_required"):
                st.caption("The older-data policy determines the applied demand choice; AI and reviewers cannot waive it.")
            else:
                st.caption("The AI projection strategy is applied unchanged." if decision["action"] == record["applied_actions"][name] else "A reviewer changed the strategy from the original AI recommendation.")
            shift = record["overrides"]["volume_shift_pp" if name == "tsa" else "price_shift_pp"]
            if shift:
                st.caption(f"Additional reviewer growth adjustment: {shift:+.1f} percentage points in every forecast quarter.")
            if recent.get("stale"):
                st.warning("Older demand data: the longer-run choice is required.")
            with st.expander("AI reasoning and evidence"):
                st.write("AI recommendation: " + ACTION_LABELS[decision["action"]])
                st.write(decision["reason"])
                st.caption("The recommendation is a judgment to review; only explicit data-quality policies force a choice.")
                for url in recent.get("source_urls", []):
                    if url.startswith("https://"):
                        st.link_button("Observation source", url)
                st.json({"recent": recent, "longer_run": structural})
    st.markdown("### 2. Optional: adjust and recalculate")
    st.caption("If the current assumptions look reasonable, leave them as they are and continue to publication. Adjustments change model assumptions, never the source observations.")
    with st.expander("Change assumptions and create a revised draft"):
        st.write("**Recent movement, then longer-run growth** follows today's signal first and gradually reduces its influence. **Longer-run growth throughout** sets the recent movement aside in every forecast quarter.")
        if not is_reviewer():
            st.caption("Recalculation uses AI explanation calls and requires the same access code as a fresh run.")
            if st.button("Enable draft adjustments"):
                st.session_state["review_access_requested"] = True
            if st.session_state.get("review_access_requested"):
                sign_in()
        with st.form(f"review_form_{record['run_id']}"):
            actions = {}
            for name, label in (("tsa", "Demand projection strategy"), ("airfare", "Pricing projection strategy")):
                options = ["use_structural"] if name == "tsa" and record["context"].get("tsa_freshness", {}).get("review_required") else list(ACTION_LABELS)
                actions[name] = st.selectbox(label, options, index=options.index(record["applied_actions"][name]), format_func=ACTION_LABELS.get,
                                             disabled=not is_reviewer(), help="Choose how the model projects this signal. You are not editing historical data or fitted coefficients.")
            st.caption("Additional growth adjustments: 0 means no extra adjustment. For example, +1 percentage point changes an assumed 2% growth rate to 3%; it does not add 1% directly to revenue.")
            volume = st.number_input("Additional demand growth (percentage points)", -5.0, 5.0, float(record["overrides"]["volume_shift_pp"]), 0.5,
                                     disabled=not is_reviewer(), help="Added to demand growth in every forecast quarter, on top of the selected strategy.")
            pricing = st.number_input("Additional pricing growth (percentage points)", -5.0, 5.0, float(record["overrides"]["price_shift_pp"]), 0.5,
                                      disabled=not is_reviewer(), help="Added to pricing growth in every forecast quarter, on top of the selected strategy.")
            reason = st.text_area("Why are you changing these assumptions?", max_chars=1500, disabled=not is_reviewer(),
                                  placeholder="Required when you change a strategy or apply a non-zero adjustment.")
            st.caption("Recalculate uses the same source data and saves a new draft. Your earlier saved run stays in the history; the public approved forecast is unchanged.")
            recalculate = st.form_submit_button("Recalculate as new draft", disabled=not is_reviewer(), icon=":material/calculate:")
        if recalculate and is_reviewer():
            if actions == record["applied_actions"] and volume == record["overrides"]["volume_shift_pp"] and pricing == record["overrides"]["price_shift_pp"]:
                st.info("These settings already match the saved draft. No recalculation or AI call was needed.")
            else:
                try:
                    with action_slot(network=False):
                        check_working_review(record)
                        revised = build_draft(ARTIFACT_DIR, record["context"], record["agent"], previous=record,
                                              volume_shift_pp=volume, price_shift_pp=pricing, signal_actions=actions,
                                              override_reason=reason, tsa_review=record.get("tsa_stale_review"))
                        revised = explain_draft(revised, record)
                        save_draft(REVIEW_DIR, revised)
                        atomic_json(REVIEW_DIR / "latest_draft.json", revised)
                        st.session_state["review_revision"] = {"run_id": revised["run_id"], "before": record["forecast"],
                            "change_usd_millions": revised["forecast"][-1]["forecast_usd_millions"] - record["forecast"][-1]["forecast_usd_millions"]}
                        st.session_state["workspace_record"] = revised
                        st.session_state.pop("publication_notice", None)
                        close_publication()
                    st.rerun()
                except (ValueError, OSError, KeyError) as error:
                    st.error(redact(error))
    st.markdown("### 3. Approve and publish the saved draft")
    st.caption("Publishing replaces the public forecast with this saved draft. If you edited fields above, recalculate and inspect the result first. An access code and final confirmation are required every time you publish.")
    published = read_approved()
    if published and published.get("reviewed_record_sha256") == record["integrity_sha256"]:
        st.success("This exact draft is already approved and published.")
    elif st.button("Approve & publish", type="primary", icon=":material/task_alt:"):
        st.session_state["publication_pending"] = record["integrity_sha256"]
    if st.session_state.get("publication_pending"):
        publish_dialog(st.session_state["publication_pending"])
    approval_export()
    if is_reviewer():
        with st.expander("Technical details"):
            if st.session_state.get("narrative_error"):
                st.error("Explanation request: " + st.session_state["narrative_error"])
            st.json({"run_id": record["run_id"], "input_hashes": record["context"]["input_csv_sha256"], "code_hash": record["context"]["code_sha256"]})
            if st.session_state.get("private_diagnostics"):
                st.code(st.session_state["private_diagnostics"], language="text")


def workspace():
    st.subheader("Run & review")
    st.caption("Load data and generate a draft, inspect AI decisions, then review assumptions and approve publication.")
    if "pending_workspace_step" in st.session_state:
        st.session_state["workspace_step"] = st.session_state.pop("pending_workspace_step")
    st.session_state.setdefault("workspace_step", "Run forecast")
    labels = {"Run forecast": "1 · Choose a run", "Inspect draft": "2 · Inspect forecast & AI", "Review & approve": "3 · Review AI choices & publish"}
    step = st.segmented_control("Workflow", list(labels), format_func=labels.get, key="workspace_step") or "Run forecast"
    if step == "Run forecast":
        cached, fresh = st.columns(2)
        with cached, st.container(border=True):
            st.markdown("### View cached forecast")
            st.write("Open the latest approved result already stored by the app. No data retrieval or AI calls. Its cutoff and approval date remain visible.")
            if st.button("View cached forecast", type="primary", icon=":material/visibility:"):
                st.switch_page(FORECAST_PAGE)
            if st.button("Inspect cached draft", icon=":material/description:"):
                record = read_snapshot("latest_draft.json") or read_snapshot("draft.json")
                if record:
                    st.session_state["workspace_record"] = record
                    select_step("Inspect draft")
                    st.rerun()
                else:
                    st.info("No cached draft is available yet. The Approved forecast page is separate.")
        with fresh, st.container(border=True):
            st.markdown("### Load latest data & forecast")
            st.write("Refresh public sources, validate inputs, obtain AI signal recommendations and calculate a six-quarter draft. It is not published automatically.")
            if st.button("Load latest data & generate forecast", icon=":material/refresh:"):
                st.session_state["fresh_run_requested"] = True
        if st.session_state.get("fresh_run_requested"):
            st.markdown("#### Fresh forecast")
            if not is_reviewer():
                sign_in()
            else:
                st.caption("Live access enabled. Source availability and the shared AI allowance are checked during the run.")
                review = stale_review_controls()
                with st.expander("Source readiness and run allowance"):
                    show_sources()
                    try:
                        st.caption(f"AI attempts used today: {attempt_usage(REVIEW_DIR / 'llm_attempts.sqlite')} / {setting('LLM_DAILY_CALL_LIMIT', '20')}. Retries and explanations count.")
                    except sqlite3.Error:
                        st.warning("The AI allowance needs attention before execution.")
                if not setting("GEMINI_API_KEY"):
                    st.info("The live AI connection is not configured here. Cached forecasts remain available.")
                if st.button("Start fresh run", type="primary", disabled=not setting("GEMINI_API_KEY"), icon=":material/play_arrow:"):
                    run_forecast(review)
    elif step == "Inspect draft":
        inspect_draft()
    else:
        review_and_approve()


def home():
    st.subheader("From public signals to a six-quarter outlook")
    st.write("An independent forecast of Delta Air Lines passenger revenue. The system combines public financial history, passenger traffic and airfare data, with AI-assisted review and an explainable numerical model.")
    for column, title, description in zip(st.columns(3),
        ("Overview", "Approved forecast", "Run & review"),
        ("Understand the workflow and the role of AI, Python and human judgment.",
         "Explore the approved forecast, assumptions, sources, historical accuracy and uncertainty.",
         "View cached results or generate a fresh draft, inspect recommendations, adjust assumptions and approve publication.")):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
    if st.button("Run data load & forecast", type="primary", icon=":material/arrow_forward:"):
        select_step("Run forecast")
        st.switch_page(RUN_PAGE)
    st.markdown("### How the workflow works")
    stages = [
        ("1 · Gather public data", "Retrieve company filings, daily passenger throughput and monthly airfare data through tools. No observations are entered by hand."),
        ("2 · Validate the inputs", "Check gaps, unusual values, freshness and the latest reported quarter. Incomplete or unreliable inputs are flagged before forecasting."),
        ("3 · AI assesses the signals", "The LLM reviews supplied evidence and recommends whether recent movements or longer-run assumptions should guide demand and pricing. Its reasons and source evidence can be inspected."),
        ("4 · Python calculates the forecast", "An explainable model turns the selected assumptions into six quarterly revenue estimates and widening illustrative ranges. The LLM does not calculate revenue."),
        ("5 · Explain what changed", "Python compares matching quarters with the previous saved forecast and reports numerical changes. The LLM supplies a short qualitative interpretation."),
        ("6 · Review before publishing", "A person inspects the evidence, optionally revises assumptions and checks the result. Only explicit approval replaces the public forecast.")]
    for offset in (0, 2, 4):
        for column, (title, description) in zip(st.columns(2), stages[offset:offset + 2]):
            with column:
                st.markdown(f"**{title}**")
                st.caption(description)
    st.caption("Saved results need no access code. Fresh execution uses the code provided with the submission; publication asks for it again.")


st.set_page_config(page_title="Signal to Forecast | Delta passenger revenue", page_icon="✈", layout="wide")
HOME_PAGE = st.Page(home, title="Overview", default=True)
FORECAST_PAGE = st.Page(forecast_page, title="Approved forecast", url_path="forecast")
RUN_PAGE = st.Page(workspace, title="Run & review", url_path="run")
page = st.navigation([HOME_PAGE, FORECAST_PAGE, RUN_PAGE], position="top")
st.title("✈ Signal to Forecast")
st.markdown("**Delta Air Lines · Passenger revenue**")
if is_reviewer():
    with st.popover("Live access enabled", icon=":material/lock_open:"):
        st.caption("Fresh execution and draft adjustments are unlocked for this session. Publishing still requires the access code again.")
        if st.button("End live access"):
            st.session_state.pop("access_token", None)
            st.session_state.pop("review_access_requested", None)
            st.rerun()
page.run()
st.divider()
st.caption("Independent case study. Not investment advice. Not affiliated with Delta Air Lines.")

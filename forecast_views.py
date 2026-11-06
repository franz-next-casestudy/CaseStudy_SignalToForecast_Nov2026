"""Read-only forecast presentation from frozen records, without network calls."""

import copy
import json
import math
import re

import altair as alt
import pandas as pd
import streamlit as st

from forecasting import REVENUE
from forecast_workflow import seal_record, validate_record

ACTION_LABELS = {"use_recent": "Recent movement, fading to longer-run growth", "use_structural": "Longer-run growth throughout"}
MODEL_LABELS = {"seasonal_naive": "Seasonal baseline", "tsa_airfare_ridge": "Signal-driven model"}


def quarter_label(quarter):
    period = pd.Period(quarter, freq="Q-DEC")
    return f"Q{period.quarter} {period.year}"


def public_record(record, redact=lambda value: value):
    context_keys = ("cutoff_date", "prepared_at_utc", "reported_history", "evidence", "baseline", "fit", "assumptions",
                    "evaluation_snapshot", "tsa_freshness", "input_csv_sha256", "code_sha256", "quarter_verification")
    result = {key: copy.deepcopy(record[key]) for key in ("run_id", "status", "created_at_utc", "cutoff_date", "forecast",
              "changes", "applied_actions", "policy_notes", "change_narrative", "sensitivities", "approved_at_utc") if key in record}
    result["context"] = {key: copy.deepcopy(record["context"][key]) for key in context_keys if key in record["context"]}
    result["overrides"] = {key: record.get("overrides", {}).get(key, 0) for key in ("volume_shift_pp", "price_shift_pp")}
    result["agent"] = {key: copy.deepcopy(record["agent"][key]) for key in ("model", "resolved_model", "time_utc", "answer", "usage", "attempts", "prompt_sha256") if key in record.get("agent", {})}

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in ("local_file", "raw_file", "reviewer", "approval_reason")}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            text = redact(value)
            text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email removed]", text)
            return re.sub(r"(?<![A-Za-z0-9:/])(?:[A-Za-z]:[\\/]|/(?:home|Users|tmp)/)[^\s\"<>]+", "[local path removed]", text)
        return value

    return clean(result)


def deployment_snapshot(record, redact=lambda value: value):
    if record.get("status") != "approved":
        raise ValueError("Only an explicitly approved forecast can be published.")
    exported = public_record(record, redact)
    exported.update(schema_version=2, reviewer="Reviewer", approval_reason="Approved; private notes withheld",
                    reviewed_record_sha256=record.get("reviewed_record_sha256"), approved_draft_id=record.get("approved_draft_id"),
                    strict_point_in_time=False, intervals_coverage_calibrated=False, submission_ready=False,
                    private_approved_record_sha256=record["integrity_sha256"])
    exported["overrides"]["reason"] = "Private reviewer explanation withheld"
    review = record.get("tsa_stale_review")
    exported["tsa_stale_review"] = ({key: value for key, value in review.items() if key not in ("reviewer", "reason")} if review else None)
    if review:
        exported["tsa_stale_review"].update(reviewer="Reviewer", reason="Older-data review recorded; private notes withheld")
    return validate_record(seal_record(exported))


def chart_rows(record, show_baseline=False):
    history = record["context"].get("reported_history", [])[-8:]
    forecast = record["forecast"]
    periods = [pd.Period(row["quarter"], freq="Q-DEC") for row in history + forecast]
    first, last = min(periods), max(periods)
    history_map = {row["quarter"]: row[REVENUE] for row in history}
    forecast_map = {row["quarter"]: row for row in forecast}
    baseline_map = {row["quarter"]: row["forecast_usd_millions"] for row in record["context"].get("baseline", [])}
    rows = []
    for period in pd.period_range(first, last, freq="Q-DEC"):
        quarter = str(period)
        row = forecast_map.get(quarter, {})
        rows.append({"Quarter": quarter_label(quarter), "order": period.ordinal,
                     "Reported": history_map.get(quarter), "Forecast": row.get("forecast_usd_millions"),
                     "Seasonal baseline": baseline_map.get(quarter) if show_baseline else None,
                     "Lower": row.get("lower_80_usd_millions"), "Upper": row.get("upper_80_usd_millions")})
    frame = pd.DataFrame(rows)
    for column in ("Reported", "Forecast", "Seasonal baseline", "Lower", "Upper"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / 1000
    return frame


def forecast_chart(record, key):
    st.subheader("Passenger revenue outlook")
    baseline_on = st.toggle("Show seasonal baseline", value=False, key=f"baseline_{key}")
    frame = chart_rows(record, baseline_on)
    labels = frame["Quarter"].tolist()
    axis = alt.X("Quarter:N", sort=labels, axis=alt.Axis(labelAngle=0, labelOverlap="greedy"), title=None)
    series = ["Reported", "Forecast"] + (["Seasonal baseline"] if baseline_on else [])
    colors = alt.Scale(domain=series, range=["#687785", "#294C70", "#A5AFBB"][:len(series)])
    long = frame.melt(id_vars=["Quarter", "order"], value_vars=series, var_name="Series", value_name="Revenue")
    long["segment"] = long.groupby("Series")["Revenue"].transform(lambda values: values.isna().cumsum())
    lines = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
        x=axis, y=alt.Y("Revenue:Q", title="USD billions", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=colors, legend=alt.Legend(orient="bottom", title=None, columns=3)),
        detail=["Series:N", "segment:N"], order="order:Q",
        tooltip=[alt.Tooltip("Quarter:N"), alt.Tooltip("Series:N"), alt.Tooltip("Revenue:Q", title="USD bn", format=".3f")])
    band = alt.Chart(frame.dropna(subset=["Lower", "Upper"])).mark_area(color="#D9E7F5", opacity=0.8).encode(
        x=axis, y="Lower:Q", y2="Upper:Q", tooltip=["Quarter:N", alt.Tooltip("Lower:Q", title="Lower USD bn", format=".3f"), alt.Tooltip("Upper:Q", title="Upper USD bn", format=".3f")])
    boundary = alt.Chart(pd.DataFrame({"Quarter": [quarter_label(record["forecast"][0]["quarter"])]})).mark_rule(color="#8795A5", strokeDash=[4, 4]).encode(x=axis)
    st.altair_chart((band + lines + boundary).properties(height=290), alt="Eight reported quarters followed by six forecasts, with an illustrative range and forecast-start boundary")
    st.caption("Reported passenger revenue is followed by six forecast quarters. The shaded area shows an illustrative range that widens further into the future; it is not a calibrated probability interval.")
    if not record["context"].get("reported_history"):
        st.warning("Reported history was not saved in this older snapshot. Newer CSV data has not been substituted.")
    with st.expander("View quarterly figures"):
        table = pd.DataFrame(record["forecast"])
        figures = pd.DataFrame({"Quarter": table["quarter"].map(quarter_label), "Estimate (USD bn)": table["forecast_usd_millions"] / 1000,
                                "Lower (USD bn)": table["lower_80_usd_millions"] / 1000, "Upper (USD bn)": table["upper_80_usd_millions"] / 1000})
        st.dataframe(figures.round(3), hide_index=True, alt="All six quarterly estimates and illustrative ranges in USD billions")


def headline(record):
    point = record["forecast"][-1]
    left, right = st.columns([2, 1])
    with left:
        st.metric(f"Passenger revenue | {quarter_label(point['quarter'])}", f"${point['forecast_usd_millions'] / 1000:,.3f}bn",
                  delta=f"{point['revenue_growth_percent']:+.1f}% year over year", delta_color="off")
    with right:
        st.metric("Illustrative range", f"${point['lower_80_usd_millions'] / 1000:,.2f}-{point['upper_80_usd_millions'] / 1000:,.2f}bn")
        st.caption(f"Forecast cutoff: {record['cutoff_date']} | {record['status'].capitalize()}")
    if point["reference_type"] == "recursive_forecast":
        st.caption(f"The year-over-year comparison uses the earlier forecast for {quarter_label(point['seasonal_reference_quarter'])}, not a reported actual.")
    verification = record["context"].get("quarter_verification", {})
    verified = verification.get("status") == "verified" and verification.get("forecast_quarters") == [row["quarter"] for row in record["forecast"]]
    if not verified:
        st.warning("Quarter sequence follows the available filings. Latest earnings-release verification is still outstanding.")
    else:
        st.caption(f"Quarter sequence checked against Delta's release and SEC filing on {verification['as_of_date']}.")


def drivers(record, key):
    st.caption("Observed facts, AI interpretation and applied assumptions are shown separately. Source links support observations, not the AI's judgment.")
    columns = st.columns(2)
    for column, signal, title in zip(columns, ("tsa", "airfare"), ("Passenger demand · TSA", "Airfare pricing · FRED/BLS")):
        recent = record["context"]["evidence"][signal.upper() + "_RECENT"]
        structural = record["context"]["evidence"][signal.upper() + "_STRUCTURAL"]
        decision = record.get("agent", {}).get("answer", {}).get("decisions", {}).get(signal)
        with column, st.container(border=True):
            st.subheader(title)
            st.metric("Observed year-over-year movement", f"{recent['raw_growth_percent']:+.1f}%")
            st.caption(f"Latest observation: {recent['latest_observation']} | {recent.get('frequency', 'Daily' if signal == 'tsa' else 'Monthly')}")
            st.caption(recent.get("comparison", "Year-over-year comparison; full window was not saved in this legacy snapshot."))
            st.markdown("**Applied assumption**")
            st.text(ACTION_LABELS[record["applied_actions"][signal]])
            if signal == "tsa" and record["context"].get("tsa_freshness", {}).get("review_required"):
                st.caption("Strategy source: enforced older-data policy.")
            elif decision:
                st.caption("Strategy source: AI recommendation." if decision["action"] == record["applied_actions"][signal]
                           else "Strategy source: reviewer override of the AI recommendation.")
            first, last = record["forecast"][0], record["forecast"][-1]
            growth_key = "tsa_growth_percent" if signal == "tsa" else "airfare_growth_percent"
            st.caption(f"Projected growth: {first[growth_key]:+.1f}% at quarter one; {last[growth_key]:+.1f}% at quarter six.")
            if decision:
                st.markdown("**AI recommendation**")
                st.text(ACTION_LABELS[decision["action"]])
                st.text(decision["reason"])
                if decision["action"] != record["applied_actions"][signal]:
                    st.warning("Applied choice differs from the AI recommendation because of policy or reviewer intervention.")
            shift = record.get("overrides", {}).get("volume_shift_pp" if signal == "tsa" else "price_shift_pp", 0)
            if shift:
                st.caption(f"Reviewer growth adjustment: {shift:+.1f} percentage points at every horizon.")
            if recent.get("stale"):
                st.warning("Older TSA observations: longer-run volume assumption required after human acknowledgement.")
            if recent.get("clipped") or structural.get("clipped"):
                st.caption("A preset growth limit was applied; raw and applied assumptions are in the evidence.")
            for url in recent["source_urls"]:
                if url.startswith("https://"):
                    st.link_button("Observation source", url, icon=":material/open_in_new:")
            with st.expander("Evidence supplied to AI"):
                st.json({signal.upper() + "_RECENT": recent, signal.upper() + "_STRUCTURAL": structural})
    st.caption("Further out, recent movements fade toward longer-run growth unless the longer-run assumption is selected throughout.")
    narrative = record.get("change_narrative", {})
    st.markdown("**Outlook and changes**")
    comparison = record.get("changes", {})
    if comparison.get("previous_run_id"):
        basis = {"latest_saved_run": "Latest completed saved forecast", "selected_forecast": "Selected forecast"}.get(
            comparison.get("comparison_basis"), "Previous forecast; comparison selection was not recorded")
        generated = comparison.get("previous_created_at_utc")
        if generated:
            try:
                generated = pd.Timestamp(generated).tz_convert("UTC").strftime("%d %b %Y, %H:%M:%S UTC")
            except (ValueError, TypeError):
                generated = "Not recorded"
        st.caption(f"Compared with: {basis} | Generated: {generated or 'Not recorded'} | "
                   f"Status: {comparison.get('previous_status', 'Not recorded')} | "
                   f"Forecast cutoff: {comparison.get('previous_cutoff_date', 'Not recorded')}")
    if narrative.get("status") in ("complete", "fallback"):
        calculated = narrative.get("facts_source", narrative.get("source")) == "calculated_record"
        st.markdown("**Numerical summary · Python**" if calculated else "**Saved summary**")
        if narrative.get("status") == "fallback":
            st.caption("Factual summary from the calculated results · AI explanation unavailable for this run")
        for item in narrative["observed_changes"]:
            st.text(item)
        st.markdown("**AI interpretation**" if narrative.get("status") == "complete" else "**Rule-based explanation · Python**")
        st.text(narrative["interpretation"])
        if calculated:
            st.markdown("**Model limitations · Python**")
        for item in narrative["cautions"]:
            st.caption(item)
    else:
        st.warning("The post-calculation explanation is unavailable. No explanation has been invented; the numerical result is retained.")
        if record.get("agent", {}).get("answer", {}).get("summary"):
            st.text(record["agent"]["answer"]["summary"])
    changes = record.get("changes", {})
    if not changes.get("previous_run_id"):
        st.caption("No previous comparable run is recorded.")
    elif not changes.get("quarter_changes"):
        st.caption("The previous run has no matching forecast quarters; no numerical comparison is available.")
    else:
        with st.expander("Matching-quarter changes"):
            frame = pd.DataFrame(changes["quarter_changes"])
            frame["Quarter"] = frame["quarter"].map(quarter_label)
            frame["Change (USD m)"] = frame["change_usd_millions"]
            st.dataframe(frame[["Quarter", "Change (USD m)"]].round(2), hide_index=True, alt="Forecast differences in USD millions for matching quarters only")
            st.caption("Numerical differences are not exact causal dollar attribution.")


def range_performance_rows(metrics):
    """Present saved model coverage and full widths without estimating new intervals."""
    required = {"model", "horizon", "origins", "empirical_range_coverage", "mean_interval_width_usd_millions"}
    if not required.issubset(metrics.columns):
        return []
    rows = []
    for item in metrics.loc[metrics["model"] == "tsa_airfare_ridge"].to_dict("records"):
        try:
            horizon, count = float(item["horizon"]), float(item["origins"])
            coverage, width = float(item["empirical_range_coverage"]), float(item["mean_interval_width_usd_millions"])
            if (not all(math.isfinite(value) for value in (horizon, count, coverage, width))
                    or horizon not in range(1, 7) or count < 1 or not count.is_integer()
                    or not 0 <= coverage <= 1 or width < 0):
                continue
            contained = coverage * count
            if not math.isclose(contained, round(contained), abs_tol=1e-6):
                continue
            rows.append({"Horizon": f"{int(horizon)} quarter{'s' if horizon != 1 else ''} ahead",
                         "Actuals inside range": f"{round(contained)} of {int(count)}",
                         "Average full range width (USD bn)": width / 1000,
                         "horizon": int(horizon)})
        except (ValueError, TypeError):
            continue
    return sorted(rows, key=lambda row: row["horizon"])


def historical_performance(record):
    evaluation = record["context"].get("evaluation_snapshot")
    if not evaluation:
        st.info("No historical evaluation was saved with this snapshot. No metrics have been substituted from newer files.")
    elif evaluation.get("status") == "unavailable":
        st.warning("Saved historical results could not be validated. Re-run the backtest locally, then prepare a new draft to include it.")
    else:
        metrics = pd.DataFrame(evaluation["metrics"])
        metrics["Model"] = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])
        subset = metrics.loc[metrics["horizon"].isin([1, 6])]
        table = subset.pivot(index="Model", columns="horizon", values="mape_percent").rename(columns={1: "One quarter ahead (%)", 6: "Six quarters ahead (%)"})
        st.markdown("**Historical forecast accuracy — MAPE (%)**")
        st.dataframe(table.round(2), alt="Model and seasonal-baseline MAPE at one and six quarters")
        st.caption("Mean Absolute Percentage Error: the average percentage size of forecast errors. Lower is better.")
        st.caption(f"Evaluation cutoff: {evaluation['cutoff_date']} | Forecast dates per horizon: {', '.join(str(value) for value in sorted(metrics['origins'].unique()))}")
        if evaluation.get("historical_input_evidence_verified"):
            st.caption("Historical inputs: TSA pages archived before each forecast date, ALFRED airfare vintages, and financial filings available before each date. Saved source files were checked during the backtest.")
        if not evaluation.get("matches_snapshot_inputs"):
            st.warning("These saved results do not match this forecast's model code, financial data or evaluation cutoff. They are shown for reference.")
        if evaluation.get("origin_dates"):
            st.caption("Historical forecast dates: " + ", ".join(evaluation["origin_dates"]))
        range_rows = range_performance_rows(metrics)
        if range_rows:
            highlights = [f"{row['Horizon']}: {row['Actuals inside range']}" for row in range_rows if row["horizon"] in (1, 6)]
            if highlights:
                st.caption("Historical actuals inside the model's ranges — " + "; ".join(highlights) + ".")
            with st.expander("Range performance across all six quarters"):
                st.dataframe(pd.DataFrame(range_rows).drop(columns="horizon"), hide_index=True,
                             column_config={"Average full range width (USD bn)": st.column_config.NumberColumn(format="%.2f")},
                             alt="Signal-driven model range coverage counts and average full widths at each forecast horizon")
                st.caption("Width is the historical average of upper minus lower bound, not the width of today's forecast. "
                           "Consider coverage and width together: very broad ranges can contain more outcomes. "
                           "These few overlapping tests do not establish reliable 80% coverage.")
        else:
            st.caption("Range coverage and widths were not recorded in this saved evaluation.")
        with st.expander("All horizons and historical results"):
            st.dataframe(metrics[["Model", "horizon", "origins", "mape_percent", "mae_usd_millions"]].round(2), hide_index=True, alt="Saved evaluation metrics at all six horizons")
            if evaluation["details"]:
                st.dataframe(pd.DataFrame(evaluation["details"]), hide_index=True, alt="Historical forecast dates, predicted and reported outcomes")
    if evaluation and evaluation.get("historical_input_evidence_verified"):
        st.caption("This tests the numerical model with its default signal choices, not AI recommendations or reviewer overrides. Four overlapping forecast dates provide limited evidence; the ranges remain illustrative and uncalibrated. Archived pages can be several days older than their forecast date.")
    elif evaluation and evaluation.get("status") != "unavailable":
        st.warning("Historical input availability is not verified for these saved results. They do not establish a no-look-ahead backtest. AI choices and reviewer overrides are not evaluated; the sample is small and forecast windows overlap.")
    with st.expander("Sensitivity of the sixth-quarter estimate"):
        scenarios = record.get("sensitivities", [])
        if scenarios:
            frame = pd.DataFrame(scenarios)
            frame["Estimate (USD bn)"] = frame["forecast_usd_millions"] / 1000
            frame["Change (USD bn)"] = frame["change_usd_millions"] / 1000
            st.dataframe(frame[["scenario", "Estimate (USD bn)", "Change (USD bn)"]].rename(columns={"scenario": "Scenario"}).round(3), hide_index=True, alt="Read-only demand and pricing sensitivity scenarios")
            demand = frame.loc[frame["scenario"].str.startswith("Demand"), "change_usd_millions"].abs().max()
            pricing = frame.loc[frame["scenario"].str.startswith("Pricing"), "change_usd_millions"].abs().max()
            if pd.notna(demand) and pd.notna(pricing):
                stronger = ("Demand has the larger effect in these scenarios." if demand > pricing else
                            "Pricing has the larger effect in these scenarios." if pricing > demand else
                            "Both have the same maximum effect in these scenarios.")
                st.caption(f"Persistent ±1 percentage-point growth adjustments change the sixth-quarter estimate by up to "
                           f"USD {demand:,.0f}m for demand and USD {pricing:,.0f}m for pricing. {stronger} "
                           "These scenarios adjust growth assumptions throughout the forecast; they do not represent a single daily TSA movement.")
        else:
            st.info("Sensitivities were not saved with this older snapshot.")
        st.caption("Hypothetical persistent ±1 percentage-point growth changes, calculated with the same model. These scenarios are separate from ranges and never change the saved forecast.")


def latest_run(record, latest_attempt=None):
    st.markdown("**Run underlying this forecast**")
    st.caption(f"Generated: {record.get('created_at_utc', 'Not recorded')} | Approved: {record.get('approved_at_utc', 'Not approved')}")
    st.text(f"Status: {record['status'].capitalize()} | Forecast cutoff: {record['cutoff_date']}")
    narrative_status = record.get("change_narrative", {}).get("status")
    explanation_role = ("The written interpretation was produced by AI." if narrative_status == "complete" else
                        "The explanation is a rule-based Python fallback; AI interpretation was unavailable." if narrative_status == "fallback" else
                        "No completed AI interpretation is recorded.")
    st.caption("Roles in this run: Python retrieves and validates data and calculates revenue; AI recommends signal strategies. "
               + explanation_role + " A person reviews overrides and publication.")
    for signal, label in (("TSA_RECENT", "Passenger demand"), ("AIRFARE_RECENT", "Airfare pricing")):
        facts = record["context"]["evidence"][signal]
        st.caption(f"{label}: observation {facts['latest_observation']} | Retrieved {facts.get('retrieved_at_utc', 'Not recorded')} | {'Older-data review required' if facts.get('stale') else 'Within the recorded model freshness policy'}")
    if latest_attempt:
        st.markdown("**Latest execution attempt**")
        st.text(f"{latest_attempt.get('time_utc', 'Unknown time')} | {latest_attempt.get('status', 'Unknown')}")
        st.text(latest_attempt.get("message", "No public summary recorded."))
        if latest_attempt.get("run_id") != record["run_id"]:
            st.caption("This is separate from the run underlying the displayed forecast; an unapproved or failed attempt does not replace approval.")
    with st.expander("Technical details"):
        st.json({"run_id": record["run_id"], "input_hashes": record["context"].get("input_csv_sha256"),
                 "agent": record.get("agent"), "policy_notes": record.get("policy_notes", []),
                 "narrative_status": record.get("change_narrative", {}).get("status", "not_generated"),
                 "explanation_diagnostics": record.get("change_narrative", {}).get("diagnostics", [])})
        st.download_button("Download public run record", json.dumps(record, indent=2), file_name=f"forecast_{record['run_id']}.json",
                           mime="application/json", icon=":material/download:")


@st.dialog("Methodology", width="large")
def methodology(record=None):
    st.markdown("""### AI, calculations and human review

| Role | Responsibility |
| --- | --- |
| Python | Retrieves sources, checks data quality, computes drivers, fits the model and calculates estimates, ranges and numerical changes. |
| AI assessment | Reviews supplied evidence and chooses a projection strategy for passenger demand and airfare pricing, with reasons and evidence references. |
| AI interpretation | Explains the calculated outlook in words. Python supplies the displayed amounts, comparisons and caveats. |
| Human reviewer | Inspects the evidence, accepts or overrides choices, reviews the recalculated result and explicitly approves publication. |

**What can AI choose?** Separately for TSA demand and airfare pricing, it can select **recent movement, fading to longer-run growth**, or **longer-run growth throughout**. These choices feed the numerical model and can change the forecast. The assessment considers recent growth, freshness, missing data, assumption limits, disagreement between signals and the supplied backtest caveats. It receives extracted facts, evidence IDs and source links; it does not independently browse or verify those links.

**What remains controlled by Python?** AI cannot edit source observations, fitted coefficients, growth limits or calculated revenue, and cannot publish a result. Older TSA data forces the longer-run demand strategy regardless of an AI or reviewer preference. Other preset bounds also remain enforced.

**How can a person change it?** Open **Run & review → Review AI choices & publish**. The cards separate the AI recommendation from the applied strategy. Keep the choice, replace its strategy, or add a bounded demand/pricing growth adjustment in percentage points. Enter a reason and select **Recalculate as new draft** to inspect the effect using the same source data. Earlier runs remain in the history. Unsaved edits are not included in publication.

**How is publication controlled?** **Approve & publish** previews the exact saved draft and its assumptions. A reviewer re-enters the access code and confirms the limitations before replacing the public approved forecast. Generating or recalculating a draft does not publish it. If the AI explanation fails, the app labels a factual Python summary as the fallback rather than presenting it as AI output.

### Numerical method""")
    st.markdown("""**Calculation.** The seasonal anchor is revenue in the same quarter a year earlier. The model fits year-over-year log revenue growth to TSA passenger-volume growth and airfare CPI growth, using non-negative Ridge coefficients. It trains on the last 12 complete joint quarters from 2022 onward, with at least eight required.

**Why these signals.** TSA counts are a daily proxy for US air-travel demand; BLS airfare CPI via FRED is a monthly proxy for market pricing. Neither is Delta-specific. International exposure, market share, loyalty revenue, route mix and premium cabins can break the relationship.

**Later quarters.** Recent growth fades linearly toward a bounded median historical growth assumption over six horizons; the fitted intercept fades to zero. A structural-only selection uses longer-run growth throughout. Volume structural growth is bounded to −3%/+6%, pricing to −3%/+4%; recent seeds are bounded to −20%/+20% and annual revenue growth to −30%/+30%.

**Recursion.** Quarters five and six use earlier forecasts as their year-earlier anchors, not future actuals.

**Ranges.** Half-width equals the mean of the last four known quarterly revenues × 1.28155 × max(training log-residual standard deviation, 5%) × √horizon, with a nonnegative lower bound. This is an illustrative widening rule, not calibrated 80% coverage.

**Historical evaluation.** The numerical model is compared with a seasonal baseline at historical forecast dates across six horizons. The Historical performance tab shows the results and the input-verification status saved with this forecast. AI decisions and reviewer overrides are not covered by that numerical backtest.""")
    evaluation = record.get("context", {}).get("evaluation_snapshot") if record else None
    if evaluation and evaluation.get("historical_input_evidence_verified"):
        st.markdown("**Historical data checks.** TSA inputs come from captures made before each forecast date; airfare uses ALFRED vintages from the previous calendar day. Financial inputs are restricted to earlier filings, with saved source files verified during the backtest. Captures are a conservative information set and may omit more recent information already available at the time. Only four overlapping forecast dates are tested; the ranges are not calibrated.")
    if record:
        history = record["context"].get("reported_history", [])
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True, column_config={"source_url": st.column_config.LinkColumn("SEC filing"),
                         "additional_source_url": st.column_config.LinkColumn("Additional source")},
                         alt="Frozen reported-history values and SEC filing source links")

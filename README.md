# Signal to Forecast

Delta Air Lines passenger-revenue forecasting case study. The application refreshes public sources, asks an LLM to assess demand and pricing evidence, calculates six quarterly forecasts, and requires human review before publication.

Independent case study. Not investment advice. Not affiliated with Delta Air Lines. Only public data is used.

## Start locally

Run commands from the project directory. Python 3.11–3.12 is recommended; the clean setup is checked on Python 3.12.

```powershell
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep these application files together: `app.py`, `forecast_views.py`, `forecast_workflow.py`, `forecasting.py`, `data_retrieval.py`, `run_controls.py`, and `verify_quarters.py`. Historical reconstruction also needs `download_airfare_vintages.py`, `download_tsa_vintages.py`, and `tsa_access_probe.py`. Include the tests and the light-theme `.streamlit/config.toml`.

The application reads local `.env` settings automatically. Example values:

```dotenv
APP_PASSWORD=replace-with-a-private-reviewer-password
GEMINI_API_KEY=replace-with-your-private-key
GEMINI_MODEL=gemini-3.5-flash-lite
SEC_NAME=Your Name
SEC_EMAIL=your-contact-email@example.com
LLM_DAILY_CALL_LIMIT=20
```

Streamlit secrets override process environment variables, which override `.env`. Hosted Streamlit secrets use the same keys as quoted, top-level TOML values. No secrets are needed to view an already published snapshot. CLI retrieval does not load `.env`; set SEC identification in the terminal that runs it:

```powershell
$env:SEC_NAME = "Your Name"
$env:SEC_EMAIL = "your-contact-email@example.com"
# macOS/Linux: export SEC_NAME="Your Name" SEC_EMAIL="your-contact-email@example.com"
```

## Retrieve data and run a forecast

```powershell
python data_retrieval.py --download-sec
python verify_quarters.py
python -m streamlit run app.py
```

Default retrieval downloads SEC filings, TSA daily throughput and FRED/BLS airfare CPI into `input_data/`. TSA uses `curl_cffi`; complete validated historical years are reused, and the current year is refreshed. Source hashes, retrieval times and original responses are retained locally. These are automatically generated runtime files, not hand-pasted inputs or datasets to commit to Git.

TSA and FRED need no API keys. SEC requires an identifying name and email; the AI assessment requires a Gemini key with available quota. Fuel is not displayed in the app, is not a forecast input and is excluded from both the default CLI and app refresh. Its optional legacy downloader remains available only through an explicit `--sources eia` command. Explicit combined runs can use `--require-eia` to make a fuel failure blocking.

Official TSA access succeeded on the development Windows machine, including a clean historical download. A separate fresh Linux-host check returned HTTP 403 despite the same client; its stale/incomplete backup was correctly rejected by strict checks. Access is network-dependent, and hosted source access remains unverified. The approved public overview itself does not request TSA or invoke the LLM.

Useful source-specific checks:

```powershell
python data_retrieval.py --sources tsa --output-dir tsa_clean_test
python data_retrieval.py --sources tsa --require-fresh-tsa --require-complete-tsa
```

In the app:

1. **Overview** explains the case and the six stages of the workflow; it has no forecast figures. **Approved forecast** contains the approved results. **Run & review** contains execution, inspection and publication.
2. Select **Run data load & forecast**, then choose **View cached forecast** (no network or AI calls) or **Load latest data & generate forecast**. For a fresh run, enter the submission access code when prompted, then select **Start fresh run**. The API key remains server-side.
3. Inspect the unapproved draft, observed signals, AI recommendations, source links and historical comparison. Select **Review AI choices** to continue.
4. Review the current strategies and their evidence. Optionally expand the adjustment form, choose recent/fading versus longer-run growth, and apply additional percentage-point shifts with a reason. **Recalculate as new draft** uses the same data, preserves the earlier run and shows the numerical effect. It never publishes automatically. Recalculation requires live access; cached inspection does not.
5. **Approve & publish** opens a confirmation of the exact saved draft and its assumptions. Re-enter the access code, add a reviewer label and approval note, and acknowledge the limitations. **Confirm publication** replaces the forecast on the public **Approved forecast** page. Unsaved form edits are not published. Anyone with the code may publish; there is no separate owner permission role.

New runs compare with the latest completed saved forecast, selected by generation time across saved draft/approval records. A later approval of an older forecast does not make it a newer numerical run. The comparison timestamp and status are displayed; **Compare with approved forecast** remains separate.

Public visitors can view approved results and inspect an available cached draft without signing in. Authentication is requested in context for live execution or recalculation, and again at every publication. A fresh checkout includes the explicitly approved submission snapshot for public viewing without live retrieval or AI calls. A concurrent change to the working-review record blocks recalculation/publication until the reviewer refreshes and inspects the review again.

## Reproduce the historical numerical backtest

First retrieve the SEC history as above. Then reconstruct the four historical signal information sets:

```powershell
python download_airfare_vintages.py
python download_tsa_vintages.py --cutoffs 2024-09-01 2024-12-01 2025-03-01 2025-06-01
python forecasting.py --backtest-only --cutoff 2026-11-06 --output-dir output_data/vintage_backtest
```

The vintage downloaders require network access. The backtest command is offline and makes no LLM request, approval or publication. Do not substitute today's signal CSVs if an archive cannot be retrieved.

ALFRED snapshots use the previous calendar day's vintage. TSA observations are checked against preserved pre-cutoff Wayback captures. Archive requests are sequential with a default 30-second pause after each response. A 429 stops the downloader and saves a cooldown; respect that cooldown, then rerun the same command to resume. Completed sources and indexes are reused. Archive access can be slow or unavailable; a stopped reconstruction is not a successful backtest.

`driver_run.json` records cutoff conventions, source verification, training windows and hashes. A successful historical-input audit requires the saved financial source documents and extraction evidence, not merely dated CSV rows. TSA captures provide a conservative information set and can precede a forecast date by several days. Financial history uses stored original-filing values, rather than a complete history of every earlier earnings announcement or later restatement.

The app loads these results from `output_data/vintage_backtest/`, checks their model code, financial inputs, evaluation date and scored predictions, then freezes the evaluation into a new draft. Historical evaluation and live forecast dates are separate: a saved evaluation can accompany a later forecast when the numerical code and complete financial model inputs still match. The UI shows its original evaluation cutoff and age; it never presents reuse as a newly executed backtest. An evaluation dated after the forecast is incompatible. Older exploratory current-vintage evaluations retain their stricter identical-date and input-file requirements. Existing approved snapshots retain their reviewed evaluation. A model/history change requires a matching backtest; a display-only change or a new calendar day alone does not. New or revised financial quarters require an updated evaluation rather than an automatic match.

For the reviewed run with evaluation cutoff 6 November 2026:

| Model | One-quarter MAPE | Six-quarter MAPE |
| --- | ---: | ---: |
| Seasonal baseline | 2.08% | 10.55% |
| TSA/airfare Ridge model | 5.63% | 5.30% |

Four origins yield 24 scored forecasts per model, with four observations at each horizon. Windows overlap. These results evaluate fixed default numerical choices, not historical LLM recommendations or reviewer overrides. Ranges contained actuals in 2/4 one-quarter cases and 4/4 six-quarter cases; this does not establish calibrated 80% coverage. Results from a new reconstruction must be read from its generated audit and metrics rather than assumed identical.

## Model and guardrails

The seasonal anchor is passenger revenue in the same quarter a year earlier. A nonnegative Ridge regression (`alpha=1`) fits year-over-year log revenue growth to TSA volume growth and airfare CPI growth, using the latest 12 complete joint quarters from 2022 onward, with at least eight required. Incomplete quarters are excluded; source blanks, including the October 2025 airfare observation, are not interpolated.

Recent TSA growth compares 28 days with the same weekdays 364 days earlier. Pricing uses three reported monthly year-over-year pairs. Recent growth fades toward bounded longer-run median growth across six horizons, while the fitted intercept fades to zero. The LLM chooses recent/fading or longer-run growth throughout for each signal; Python computes the numbers. Quarters five and six recursively use earlier forecasts as their seasonal anchors.

Recent growth is bounded to −20%/+20%; longer-run demand to −3%/+6%, pricing to −3%/+4%, and revenue growth to −30%/+30%. Reviewer adjustments are bounded to ±5 percentage points and require a reason. TSA observations older than seven days require explicit review and longer-run demand assumptions. Missing, stale or invalid required inputs remain subject to validation.

Range half-width is the mean of the last four known quarterly revenues × 1.28155 × max(training log-residual standard deviation, 5%) × √horizon. The uncertainty floor and widening rule are simplifying judgments; ranges are illustrative, not calibrated probability intervals. The app shows observed historical coverage and average full width at all six horizons.

Read-only sensitivities apply persistent ±1 percentage-point demand/pricing growth adjustments throughout the forecast. They are separate from uncertainty ranges and from a single daily signal movement. TSA and airfare CPI are market-wide proxies: Delta's network, capacity, cabin mix, international exposure and market share can diverge from them. Fuel is excluded from the app and default data intake; it has no effect on forecasts.

Methodology documents the AI choices, Python calculations and human override/publication flow. The UI labels the numerical summary, AI interpretation and applied strategy source separately. The post-calculation LLM supplies only a short qualitative interpretation. Python writes all displayed amounts, comparisons and caveats. An invalid or failed explanation is retried within the allowance; if it remains unavailable, a clearly labelled factual summary preserves the numerical draft. The system never invents a successful AI assessment.

After each local backtest, run `python annotate_backtest_financials.py` before uploading its results. This verifies that the local financial CSV exactly matches the original backtest hash, then adds a separate identity for the complete revenue history, quarter dates, filing dates and SEC filing links. It does not change historical predictions, metrics or source hashes. Cloud comparisons can then distinguish identical model inputs from changes to downloaded source-document bytes. Any changed revenue, date or filing link still produces an evaluation mismatch. The original source verification remains specific to the saved historical evaluation, not a claim that fresh response bytes are identical.

## Tests and run controls

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

Tests use synthetic inputs and mocked providers; they are not the submitted forecast or proof of hosted source access. Every Gemini attempt is reserved in SQLite before sending. Each request phase permits at most one retry for recoverable server, rate-limit, network or response failures. Retry hints are respected within a bounded wait; permanent errors are not repeatedly retried. AI interpretation validation can also trigger the bounded retry.

The live-run cooldown is 60 seconds within one process. The local SQLite allowance is not a durable Cloud-wide spending cap and can reset if the hosting disk is replaced. Anonymous live AI execution is disabled. There is no paid-provider fallback. Reviewer labels are self-reported; the password gate is suitable only for this public-data case study, not sensitive production data.

## Runtime files and deployment status

Keep raw `input_data/`, working run records, probe/test output directories, `.venv/`, `.env` and `.streamlit/secrets.toml` outside Git. Include code, tests, dependencies and `.streamlit/config.toml`. The submission intentionally includes one redacted approved snapshot at `output_data/published/approved.json` and four evaluation outputs under `output_data/vintage_backtest/`: `driver_run.json`, `model_comparison.csv`, `baseline_backtest.csv` and `driver_backtest.csv`. Include `annotate_backtest_financials.py` with the other application and reconstruction scripts. Check both ignore rules and tracked files: adding an ignore rule does not untrack existing files.

The overview reads a privacy-redacted approved snapshot from `output_data/published/approved.json` under the runtime artifact root. `FORECAST_ARTIFACT_DIR` can change that root through the process environment. The exported snapshot contains the forecast, sources, narrative and historical evaluation needed for public viewing; raw intake files are needed for live updates, not for rendering that approved overview.

The committed approved snapshot supplies the public submission forecast on a fresh deployment without downloading data or calling AI. The four committed evaluation outputs let newly generated drafts include matching historical results. Cloud publication updates runtime files, not GitHub: retaining a newer approval across rebuilds requires replacing the committed deployment export manually. External persistent storage is not implemented. Do not regenerate or auto-approve a replacement at startup.

Before submission: personally approve the final reviewed draft; verify the public overview in a signed-out browser after a cold restart; test hosted source access and the permitted run path; recheck the latest reported quarter on submission day; and include the repo/ZIP, CFO memo, build log and insurer-adaptation document. Record actual build time honestly.

Next improvements would be broader historical evaluation and independent interval calibration, stronger signal-choice evaluation, and durable storage with authenticated reviewer identities. They are not claims about the current build.

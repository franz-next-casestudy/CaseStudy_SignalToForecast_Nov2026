# Delta Data Retrieval

One self-contained Python 3.10+ script downloads SEC documents, extracts Delta
financials, refreshes TSA/EIA/FRED signals, and saves validated CSVs with provenance.
The separate `app.py` provides a password-protected deployment test. The original
app in `0_TestDeployment` remains unchanged.

## Test Deployment App

From this folder, install dependencies and start the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The local `.env` file has blank credential fields and is ignored by Git. Fill in
the following values privately; restart the server after changing the file.
Existing process environment variables take precedence over `.env`, and
Streamlit secrets take precedence over both. The app loads `.env` automatically;
the standalone retrieval CLI still expects SEC identity in its process environment.

| Variable | Required For | Value |
| --- | --- | --- |
| `APP_PASSWORD` | Opening the dashboard | A long, unique random password, preferably 20+ characters |
| `GEMINI_API_KEY` | Gemini test | Your Google AI Studio API key |
| `GEMINI_MODEL` | Optional | Defaults to `gemini-3.8-flash`; use a model ID available to your key and free-tier quota |
| `SEC_NAME` | Downloading SEC filings | Your real name or organization identifying the request |
| `SEC_EMAIL` | Downloading SEC filings | A valid contact email for SEC identification |

TSA, EIA and FRED downloads do not need API keys. Saved CSVs can be inspected
without SEC identity or a Gemini key. To rebuild financial CSVs without an SEC
download, the local SEC manifest and original documents must also exist.

On Streamlit Community Cloud, select this folder's `app.py` as the entry point
and paste the following top-level TOML entries into the app's **Secrets** settings,
replacing the placeholders privately. Do not upload `.env` or commit a secrets file.

```toml
APP_PASSWORD = "your-long-unique-password"
GEMINI_API_KEY = "your-google-ai-studio-key"
GEMINI_MODEL = "gemini-3.8-flash"
SEC_NAME = "your-name-or-organization"
SEC_EMAIL = "your-contact-email"
```

Include the saved `input_data/*.csv` files in the deployment for immediate
inspection. SEC originals are ignored by Git; on a fresh Cloud instance, keep
**Refresh SEC filings** selected when refreshing financials. Hosting access to
SEC/TSA must be tested separately; local success does not guarantee Cloud access.
If SEC is blocked, stop retrying and keep the saved CSVs for inspection.

The app is titled **Test Deployment**, with **Data & API Connections** shown
after unlocking. It offers selected-source refresh, CSV coverage/freshness/value
and provenance checks, recent rows, a history chart, and a Gemini connection test.
The displayed checks are a lightweight inspection, not the full retrieval
validation or a point-in-time audit. Refreshing runs the existing retrieval
validator. Failed retrievals retain the existing CSVs; a hard timeout may leave
partial raw downloads, so inspect the data before retrying.

Gemini receives only a compact summary of public CSV observations, units and
quality flags. It is asked which dataset has the newest observation and which
caveats need review. The app validates JSON structure, not the factual accuracy
of its prose; this is a connection test, not yet a forecasting agent. No automatic
API call is made on page load. Invalid keys, model access, quota errors and
timeouts are surfaced without exposing the key or retrying automatically.
Transient server errors (HTTP 500, 502, 503 or 504) receive one immediate retry
only, with a maximum of two attempts per click. Persistent errors stop the test;
no alternate model is selected automatically. Successful response logs include
the attempt count. A 503 indicates temporary provider unavailability or overload,
not evidence that the API key is invalid.
Free-tier availability depends on the selected model/account; no billing is needed
for the other sources.

Security and operational limits:

- Access fails closed when `APP_PASSWORD` is blank. The password is checked
  server-side, removed from widget state after submission, and never logged.
- Five failed attempts pause that browser session for one minute. This is a
  temporary shared-password gate, not individual identity, MFA or robust
  brute-force protection. Use platform access controls or OIDC for sensitive data.
- Network actions are serialized within one server process, with a one-minute
  cooldown per action across sessions. This is not a durable quota or spending cap;
  use provider quotas and do not share the password publicly.
- Signal requests time out after 30 seconds each; an entire refresh is capped at
  ten minutes. Each Gemini attempt has a 45-second read timeout and a bounded
  output size; a transient-error retry can extend the overall wait.
- Run logs hold the last 20 actions in the current session, not a persistent audit
  trail. Logs redact configured credentials. Sign-out clears session results.
- Cloud disk, caches and sessions are ephemeral. Downloads may need repeating
  after a restart. No forecast or approval/publication workflow exists yet.

## Run

From this folder, install dependencies once:

```powershell
python -m pip install -r requirements.txt
```

For all four datasets, use:

```powershell
python data_retrieval.py
```

The default pipeline reads Delta from the local `input_data/sec/filing_manifest.csv`
and refreshes TSA, EIA, and FRED over the network. To process only Delta, pass
`--sources delta`; this requires no network requests or credentials.

To download or refresh SEC documents before generating CSVs:

```powershell
python data_retrieval.py --download-sec
```

Configure `SEC_NAME` and `SEC_EMAIL` in that same terminal first. The script sends
them to SEC as the required User-Agent but does not print them or save them in
project outputs. These are identity values, not API keys. For a small SEC-only
access test, run `python data_retrieval.py --sec-test`; it requests one recent
10-Q and one 10-K filed in 2020. Cached files do not prove a new document download.
Do not repeatedly retry if SEC blocks access.

Use `--start-year` / `--end-year` to restrict reporting years (default 2019 through
the current year), `--sources delta tsa eia airfare` to select datasets, `--output-dir`
to change the output location, and `--timeout` for signal-request timeouts.
SEC downloads also include the following filing year where available so the
selected final year's annual report can be retrieved. Requests are sequential.
Exit code 0 means the selected workflow succeeded, 1 means retrieval or validation
failed, and 2 means invalid command-line arguments. Output CSVs are replaced only
after the selected datasets pass validation.

## Outputs

- `input_data/delta_quarterly.csv`: passenger revenue (USD millions), RPM and
  ASM (millions), yield (cents), and load factor (percent), plus SEC source links.
- `input_data/delta_evidence.csv`: join on `quarter` and `result_metric` for
  per-value source links, accession numbers, filing dates, SEC acceptance times
  as supplied in Eastern time, local document hashes, XBRL contexts or table
  labels, result values, formulas, and source input values.
- `input_data/tsa_daily.csv`: daily passenger counts with source URLs and retrieval metadata.
- `input_data/jet_fuel_weekly.csv`: weekly USD-per-gallon prices with source URLs and retrieval metadata.
- `input_data/airfare_cpi_monthly.csv`: BLS airline-fares CPI via FRED, series
  `CUSR0000SETG01`, monthly and seasonally adjusted, index 1982-1984=100. Includes
  the series-page URL, download URL, retrieval time, source hash, units, and
  value status. Source blanks remain empty with `missing_at_source`; no values
  are interpolated. To refresh only this source, run:

  ```powershell
  python data_retrieval.py --sources airfare
  ```
- `input_data/retrieval_log.csv`: diagnostics for signal download attempts.
- `input_data/sec`: original submissions, primary documents, and a CSV manifest.
- `input_data/raw`: hashed snapshots of downloaded signal responses.

Existing `delta_source_check` snapshots are preserved for reference but are no
longer used by the script. SEC documents and old probe snapshots remain ignored
by Git. Manifest inventory timestamps are not SEC publication dates.

## Provenance And Validation

In the app, show the direct SEC `source_url` beside a reported metric. For a
derived figure, show the formula and all distinct input links from the evidence
CSV. Q4 revenue is annual minus reported nine-month revenue; Q4 RPM and ASM are
annual minus the first three quarters. Q4 yield and load factor are recomputed
from their components, never obtained by subtracting annual ratios.

Revenue extraction requires the exact `us-gaap:PassengerMember` dimension; it
does not confuse passenger ticket or loyalty components with total revenue.
Operating metrics require consolidated table headers and verified units.
Validation checks complete quarterly coverage, positive metrics, source evidence,
load factors, and traffic/yield identities with published-value rounding tolerance.
The 2025 first-three-quarter revenue sum exceeds the nine-month figure by USD
1 million. That difference is recorded explicitly; discrepancies greater than
USD 2 million stop extraction.

`published_date` is the conservative SEC filing date, not the earnings-release
date. Intraday cutoffs must use SEC acceptance times, not assume a filing was
available at the start of its filing day. This is a first-disclosure dataset,
not a full history of restatement vintages. Signal publication dates and revisions
still need separate validation for a no-look-ahead backtest. The latest reported
quarter must also be checked against earnings releases before submission.

## Sources

- [SEC EDGAR](https://www.sec.gov/edgar/browse/?CIK=27904): Delta 10-Q/10-K filings.
- [TSA](https://www.tsa.gov/travel/passenger-volumes): current and yearly passenger
  tables. Only each observation's own-year count is used, not comparison columns.
- [EIA XLS](https://www.eia.gov/dnav/pet/hist_xls/EER_EPJK_PF4_RGC_DPGw.xls): weekly
  US Gulf Coast jet fuel spot prices. Fuel is an indirect revenue signal.
- [FRED airfare CPI](https://fred.stlouisfed.org/series/CUSR0000SETG01): monthly
  market-pricing proxy, not Delta's yield or an actual ticket price. The script
  uses FRED's date-bounded CSV download endpoint with `requests`; no API key is
  required. It retrieves the latest vintage, not historical ALFRED vintages.

Current TSA data must be no more than 7 days old; fuel data no more than 14 days.
The latest nonmissing airfare observation month must be no more than 95 days
old. Monthly rows must be continuous; explicitly blank source values are allowed
and flagged. FRED dates label the observation month, not its release date, and
seasonally adjusted history can be revised. In particular, retain the October
2025 source blank rather than treating it as zero. Point-in-time airfare backtests
still require historical release/vintage handling.
These are operational checks, not proof of publication timing. Historical signal
revisions and release dates remain unverified. Local access also does not prove
access from Streamlit Community Cloud.

The script does not bypass access controls, prove reuse permissions, validate
predictive value, or certify a no-look-ahead backtest.
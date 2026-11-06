"""Add a model-input identity to a saved audit only after verifying its original CSV.

Run after a local backtest: python annotate_backtest_financials.py
No forecasts, historical metrics, source hashes or approvals are changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from forecast_workflow import atomic_json, financial_model_reference


def annotate(financial_path, audit_path):
    financial_path, audit_path = Path(financial_path), Path(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    original_hash = hashlib.sha256(financial_path.read_bytes()).hexdigest()
    if audit.get("input_csv_sha256", {}).get("delta_quarterly.csv") != original_hash:
        raise ValueError("CSV differs from the backtested original. Re-run the backtest before annotating.")
    reference = financial_model_reference(financial_path)
    reference["original_csv_sha256"] = original_hash
    reference["basis"] = "Validated revenue values, quarter dates, filing dates and SEC filing links; source-byte hashes retained separately."
    audit["financial_model_reference"] = reference
    atomic_json(audit_path, audit)
    return reference


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financials", type=Path, default=base / "input_data" / "delta_quarterly.csv")
    parser.add_argument("--audit", type=Path, default=base / "output_data" / "vintage_backtest" / "driver_run.json")
    args = parser.parse_args()
    reference = annotate(args.financials, args.audit)
    print(f"Verified original CSV and recorded {len(reference['rows'])} financial quarters. Historical results and original hashes retained.")

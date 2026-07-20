#!/usr/bin/env python3
"""Fail-closed publication gate for the 68AM clean-room evidence package."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/full_evidence.json"


def main() -> None:
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "repro/tests", "-v"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "repro/src/verify_iccl.py", "--output", "outputs/full_evidence.json"], cwd=ROOT, check=True)
    record = json.loads(EVIDENCE.read_text())
    claims = record["claims"]
    assert claims["claim_1_interference"]["uniform_predictions_changed"] == 500
    assert claims["claim_1_interference"]["causal_prefix_predictions_changed"] == 0
    assert claims["claim_2_positive_negative_transfer"]["positive_transfer_cells"] > 0
    assert claims["claim_2_positive_negative_transfer"]["negative_transfer_cells"] > 0
    assert claims["claim_3_order_and_long_prompt"]["task_permutations"] == 120
    assert claims["claim_3_order_and_long_prompt"]["fixed_prefix_permutation_invariant"] is True
    assert record["independent_formula_audit"]["max_expanded_vs_independent_error"] < 1e-11
    assert record["independent_formula_audit"]["negative_control_literal_T_plus_M_plus_1_failures"] > 0
    assert record["monte_carlo_readback"]["total_trials"] == 720_000
    assert record["monte_carlo_readback"]["maximum_absolute_z_score"] < 4.0
    marker = ROOT / ".trackio/logbook/pages/conclusion/page.md"
    assert marker.exists() and "FULL_GATE_READY: 68AMoK2YNk" in marker.read_text()
    gate = {"paper": "68AMoK2YNk", "status": "passed", "tests_passed": True, "publication_gate_passed": True,
            "source_tar_sha256": record["source"]["tar_sha256"], "evidence_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
            "claim_outcomes": {"claim_1_interference": "verified", "claim_2_transfer": "verified", "claim_3_order_long_prompt": "verified"},
            "scope_disclosures": record["scope_disclosures"]}
    (ROOT / "outputs/PUBLICATION_GATE_PASSED.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__": main()

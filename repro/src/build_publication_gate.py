#!/usr/bin/env python3
"""Build the scoped publication gate for the ICML 2026 ICCL audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPRO_OUT = ROOT / "repro" / "outputs"
OUT = ROOT / "outputs"
SOURCE_SHA256 = "026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9"
THEORY_REPORTS = (
    "claim1_generalization.json",
    "claim2_forgetting.json",
    "claim3_nonmonotone.json",
    "claim4_asymptotic.json",
)


def digest(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def load(file_name: str) -> dict:
    return json.loads((REPRO_OUT / file_name).read_text(encoding="utf-8"))


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "repro/tests", "-v"],
        cwd=ROOT,
        check=True,
    )

    source_path = ROOT / "source" / "main.tex"
    if digest(source_path) != SOURCE_SHA256:
        raise SystemExit("source/main.tex hash does not match the pinned source")

    reports = {name: load(name) for name in THEORY_REPORTS}
    if any(report.get("verdict") != "VERIFIED" for report in reports.values()):
        raise SystemExit("one or more theorem reports is not VERIFIED")
    if reports[THEORY_REPORTS[0]]["route_C_closed_vs_mc_sweep"]["sweep_configs"] != 360:
        raise SystemExit("C1 scope changed")
    if reports[THEORY_REPORTS[1]]["route_B_exhaustive_signs"]["total_(T,t,M)_checked"] != 13650:
        raise SystemExit("C2 scope changed")
    if reports[THEORY_REPORTS[2]]["sweep"]["total_configs"] != 780:
        raise SystemExit("C3 scope changed")
    if not reports[THEORY_REPORTS[3]]["checks"]["variance_tail_slope_is_minus_one"]:
        raise SystemExit("C4 asymptotic control failed")

    gpt2 = load("claim5a_gpt2_icl.json")
    qwen = load("claim5b_qwen_realworld.json")
    if gpt2.get("verdict") != "PARTIAL":
        raise SystemExit("C5a snapshot changed; review before changing its status")
    if qwen.get("verdict") != "PARTIAL":
        raise SystemExit("C5b snapshot changed; review before changing its status")

    artifacts = {
        "source/main.tex": digest(source_path),
        **{f"repro/outputs/{name}": digest(REPRO_OUT / name) for name in THEORY_REPORTS},
        "repro/outputs/claim5a_gpt2_icl.json": digest(REPRO_OUT / "claim5a_gpt2_icl.json"),
        "repro/outputs/claim5b_qwen_realworld.json": digest(REPRO_OUT / "claim5b_qwen_realworld.json"),
    }
    gate = {
        "schema_version": 1,
        "paper": {
            "openreview": "68AMoK2YNk",
            "arxiv": "2605.28705",
            "title": "Understanding Generalization and Forgetting in In-Context Continual Learning",
            "authors": ["Guangyu Li", "Meng Ding", "Lijie Hu"],
        },
        "repository": "MachineLearning-Nerd/icml26-in-context-continual-learning",
        "status": "passed",
        "evidence_release_gate": "PASSED",
        "overall_status": "VERIFIED_SCOPED_WITH_UNREPRODUCED_EXPERIMENTS",
        "strict_paper_level_gate": "NOT_READY",
        "external_verdict": {
            "last_recorded_points": 4,
            "possible_points": 10,
            "forecast_points": 8,
            "forecast_is_official": False,
            "score_increase_claimed": False,
        },
        "claim_outcomes": {
            "c1_theorem_4_3_generalization": "VERIFIED_SCOPED",
            "c2_theorem_4_4_forgetting": "VERIFIED_SCOPED",
            "c3_nonmonotone_context_effect": "VERIFIED_SCOPED",
            "c4_asymptotic_interference_floor": "VERIFIED_SCOPED",
            "c5a_tiny_gpt2": "RETAINED_REPORT_ONLY_NOT_REGENERABLE",
            "c5b_qwen_real_world_iccl": "DIVERGENT_RETAINED_REPORT_NOT_REGENERABLE",
        },
        "local_contract": {
            "theorem_claims_verified": 4,
            "focused_tests": 4,
            "c5a_committed_snapshot": "PARTIAL_50_STEP_RUN",
            "c5b_committed_snapshot": "PARTIAL_QWEN_0.5B_32_QUERY_RUN",
        },
        "source": {
            "main_tex_sha256": SOURCE_SHA256,
            "source_tar_present": False,
            "source_tar_recorded_sha256": "c28cca01207449b7dea91544161b5f8c764ef1001f729efd17073ba7b4101210",
        },
        "artifacts": artifacts,
        "limitations": [
            "C1-C4 are finite formula and synthetic checks, not machine-checked universal proofs.",
            "The later 34k-step GPT-2 report has no committed checkpoint or raw metric table.",
            "The later Qwen2.5-1.5B report has no committed raw artifact; the committed JSON is an earlier 0.5B partial snapshot.",
        ],
    }
    encoded = json.dumps(gate, indent=2, sort_keys=True) + "\n"
    (ROOT / "publication_gate.json").write_text(encoded, encoding="utf-8")
    (OUT / "publication_gate.json").write_text(encoded, encoding="utf-8")
    (OUT / "RELEASE_CANDIDATE_READY.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ICCL reproduction run harness (arXiv:2605.28705, OpenReview 68AMoK2YNk).

This is the single fixed entrypoint launched by `orx exp run` on every node:
  command: uv run python repro/run.py

It discovers the claim verifiers committed on this branch, runs them in order,
prints an evaluator visible summary (the run log is the only evidence channel in
local mode), and writes outputs/run_summary.json.

Contract:
- A verifier that RAISES an exception is a hard failure (crash) -> the whole run
  exits nonzero. Rigorous claim verifiers assert their claim, so a claim that is
  not verified raises AssertionError and fails the run.
- A verifier may also return an integer verdict (0 = supports, nonzero =
  inconclusive/falsified); that verdict is recorded per-verifier but does not by
  itself fail the run unless it raised. This lets the baseline branch record the
  honest toy verdicts (some "inconclusive") as the 4/10 reference without masking
  crashes.

Baseline branch: runs the existing toy-scale verifiers as the 4/10 reference.
Children replace these with rigorous, full-scale verifiers (see per-branch code).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYS_SRC = REPO_ROOT / "repro" / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.insert(0, str(SYS_SRC))

# Registry of (claim_key, module_name, human_title). Edited per branch.
VERIFIERS = [
    ("claims_1_2_3_toy_reference", "verify_iccl",
     "Claims 1-3 (toy reference): interference identity, transfer, order/long-prompt"),
    ("claim_nonmonotone_toy_reference", "verify_incontext",
     "Non-monotone in-context error (toy reference, Theorem 4.3 closed form)"),
    ("claim_4_asymptotics_toy_reference", "verify_claim4_interference_asymptotics",
     "Claim 4 (toy reference): O(1/M) variance decay + persistent mean floor"),
]


def _to_jsonable(o):
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    return str(o)


def main() -> int:
    start = time.time()
    summary = {
        "paper": {"arxiv": "2605.28705", "openreview": "68AMoK2YNk"},
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "verifiers": [],
    }
    all_executed = True
    for key, mod_name, title in VERIFIERS:
        entry = {"key": key, "title": title, "module": mod_name}
        try:
            mod = importlib.import_module(mod_name)
            t0 = time.time()
            rc = mod.main()
            if rc is None:
                rc = 0
            entry["exit_code"] = int(rc)
            entry["verdict"] = "supports" if int(rc) == 0 else "inconclusive_or_falsified"
            entry["executed"] = True
            entry["seconds"] = round(time.time() - t0, 3)
        except SystemExit as e:
            code = int(e.code) if isinstance(e.code, int) else 1
            entry["exit_code"] = code
            entry["verdict"] = "supports" if code == 0 else "inconclusive_or_falsified"
            entry["executed"] = True
            entry["seconds"] = None
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=6)
            entry["executed"] = False
            entry["exit_code"] = 1
            entry["verdict"] = "crashed"
            all_executed = False
        summary["verifiers"].append(entry)

    summary["all_executed"] = all_executed
    summary["wall_seconds"] = round(time.time() - start, 3)

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_to_jsonable(summary), indent=2, sort_keys=True)
    (out_dir / "run_summary.json").write_text(payload + "\n")

    print("\n" + "=" * 72)
    print("ICCL REPRODUCTION RUN SUMMARY")
    print("=" * 72)
    for v in summary["verifiers"]:
        flag = "RAN" if v.get("executed") else "CRASH"
        print(f"[{flag}] {v['key']} (exit={v.get('exit_code')}, verdict={v.get('verdict')})")
    print("-" * 72)
    print(f"all_executed={all_executed}  wall={summary['wall_seconds']}s")
    print("RUN_SUMMARY_SHA256=" + hashlib.sha256(payload.encode()).hexdigest())
    return 0 if all_executed else 1


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

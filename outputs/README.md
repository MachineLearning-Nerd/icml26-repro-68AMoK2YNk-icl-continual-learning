# Output guide

| File | Meaning |
| --- | --- |
| `full_evidence.json` | Historical three-claim toy-scale evidence |
| `PUBLICATION_GATE_PASSED.json` | Historical three-claim baseline marker |
| `publication_gate.json` | Canonical combined C1–C5 release decision |
| `RELEASE_CANDIDATE_READY.json` | Byte-identical convenience copy of the canonical gate |

The theorem-level JSON reports live under `repro/outputs/`. The model-scale JSON
files there are branch-captured partial snapshots; they are not silently treated
as the later external runs described by the report.

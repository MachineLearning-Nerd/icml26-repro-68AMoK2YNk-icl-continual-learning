# Publication gate

The canonical gate separates local theorem evidence from retained model-scale
reports and from the old external score.

## Local decision

| Gate | Result | Interpretation |
| --- | --- | --- |
| Evidence-release gate | `PASSED` | C1–C4 pass their declared finite contracts; C5 artifact boundaries are explicit |
| Overall status | `VERIFIED_SCOPED_WITH_UNREPRODUCED_EXPERIMENTS` | The theorem package is verified within scope, while C5 is not fresh-clone complete |
| Strict paper-level gate | `NOT_READY` | Missing raw model-scale artifacts prevent a full paper reproduction claim |

## Required checks

`repro/src/build_publication_gate.py`:

1. runs the four focused unit tests;
2. requires the generated C1–C4 reports to say `VERIFIED` and checks their
   decisive finite outcomes;
3. checks the C5 snapshots are marked `PARTIAL` and records their model/artifact
   mismatch instead of upgrading them;
4. verifies the committed `source/main.tex` hash; and
5. writes identical JSON to `publication_gate.json`,
   `outputs/publication_gate.json`, and `outputs/RELEASE_CANDIDATE_READY.json`.

Run it with:

```bash
uv run python repro/run.py
python -m unittest discover -s repro/tests -v
uv run python repro/src/build_publication_gate.py
```

## External score boundary

The inventory records the last external judge score as 4/10. The report's 8/10
number is a conservative forecast after the theorem revision, not an official
rejudge. No score increase is claimed by this repository.

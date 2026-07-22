# Negative controls


---
<!-- trackio-cell
{"type": "code", "id": "cell_a67812ac964f", "created_at": "2026-07-20T08:45:29+00:00", "title": "Unit tests", "command": ["python", "-m", "unittest", "discover", "-s", "repro/tests", "-v"], "exit_code": 0, "duration_s": 0.078}
-->
````bash
$ python -m unittest discover -s repro/tests -v
````

exit 0 · 0.1s


````output
test_formula_identity (test_iccl.ICCLTests.test_formula_identity) ... ok
test_source_hash (test_iccl.ICCLTests.test_source_hash) ... ok
test_transfer_has_both_signs (test_iccl.ICCLTests.test_transfer_has_both_signs) ... ok
test_weight_forms (test_iccl.ICCLTests.test_weight_forms) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.003s

OK

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7f65d82500d0", "created_at": "2026-07-20T08:45:45+00:00", "title": "Scope controls"}
-->
The literal appendix denominator T+M+1 fails 9600 source-formula checks; TM+T+1 is required. Prediction drift is recorded separately from target-risk change. Fixed-prefix permutation invariance prevents an overclaim about arbitrary order sensitivity.

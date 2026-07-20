import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).parents[1] / "src" / "verify_iccl.py"
spec = importlib.util.spec_from_file_location("iccl", path)
iccl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(iccl)


class ICCLTests(unittest.TestCase):
    def test_source_hash(self): self.assertEqual(iccl.source_manifest(), iccl.SOURCE)
    def test_weight_forms(self): self.assertEqual(iccl.task_prediction_weights(2, 3), 1 / 8)
    def test_formula_identity(self):
        mu, var, lam = [[1., 0.], [0., 1.]], [[.2, .3], [.4, .5]], [1., 2.]
        self.assertAlmostEqual(iccl.source_interference_expanded(mu, var, lam, 4, 1), iccl.independent_interference(mu, var, lam, 4, 1), places=12)
    def test_transfer_has_both_signs(self):
        outcome = iccl.transfer_audit(); self.assertGreater(outcome['positive_transfer_cells'], 0); self.assertGreater(outcome['negative_transfer_cells'], 0)


if __name__ == '__main__': unittest.main()

import unittest
import numpy as np
from feasibility_backtracking import feasible_backtracking, needs_remesh


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.base = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.tets = np.array([[0, 1, 2, 3]])

    def test_accepts_successful_trial_without_loss_decrease(self):
        delta = np.full_like(self.base, 0.1)
        x, scale, loss = feasible_backtracking(self.base, delta, self.tets, lambda x: 100.)
        np.testing.assert_allclose(x, self.base + delta)
        self.assertEqual((scale, loss), (1., 100.))

    def test_rejects_inversion_and_degeneracy_before_solving(self):
        delta = np.zeros_like(self.base)
        delta[3, 2] = -2
        seen = []
        def evaluate(x):
            seen.append(x.copy())
            return 1.
        x, scale, _ = feasible_backtracking(self.base, delta, self.tets, evaluate)
        self.assertEqual(scale, .25)
        self.assertEqual(len(seen), 1)
        self.assertEqual(x[3, 2], .5)
        self.assertTrue(needs_remesh(.5, .01, scale))

    def test_failed_rollout_shrinks_and_exhaustion_restores(self):
        delta = np.full_like(self.base, .1)
        seen = []
        def evaluate(x):
            seen.append(x.copy())
            if not np.array_equal(x, self.base):
                raise RuntimeError('forward failed')
            return 1.
        x, scale, _ = feasible_backtracking(self.base, delta, self.tets, evaluate, max_backtracks=2)
        self.assertEqual(len(seen), 4)
        np.testing.assert_allclose(seen[1], self.base + .5 * delta)
        np.testing.assert_array_equal(seen[-1], self.base)
        np.testing.assert_array_equal(x, self.base)
        self.assertEqual(scale, 0)
        self.assertTrue(needs_remesh(1, .01, scale))

    def test_nonfinite_trial_shrinks(self):
        values = iter([float('nan'), 2.])
        _, scale, _ = feasible_backtracking(self.base, np.full_like(self.base, .1), self.tets, lambda x: next(values))
        self.assertEqual(scale, .5)
        self.assertFalse(needs_remesh(1, .01, scale))
        self.assertTrue(needs_remesh(.001, .01, scale))


if __name__ == '__main__':
    unittest.main()

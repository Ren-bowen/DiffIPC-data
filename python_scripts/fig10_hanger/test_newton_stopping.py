"""Configuration regression tests without importing the native simulator."""
import ast
import copy
import json
from pathlib import Path
import types
import unittest

import numpy as np


class NewtonStoppingTests(unittest.TestCase):
    def test_case_explicitly_uses_psd_newton_strategy(self):
        config = json.loads(Path(__file__).with_name('run.json').read_text())
        nonlinear = config['solver']['nonlinear']
        self.assertEqual(nonlinear['solver'], 'Newton')
        self.assertEqual(nonlinear['Newton'], {
            'force_psd_projection': True,
            'use_psd_projection': True,
        })

    def setUp(self):
        source = Path(__file__).with_name('optimize.py').read_text()
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('configure_newton_stopping', 'init_solver')]
        self.env = dict(np=np, copy=copy, json=json, _newton_stopping='unified-linf',
                        _legacy_polysolve_schema=True)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(__file__), 'exec'), self.env)
        self.config = {'solver': {'nonlinear': {'grad_norm_tol': 1e-3}}}

    def test_absolute_linf_no_second_bbox_or_dt_scaling(self):
        tolerance = self.env['configure_newton_stopping'](self.config, 1e-3, 0.02, 2.7)
        self.assertAlmostEqual(tolerance, 5.4e-5)
        nl = self.config['solver']['nonlinear']
        self.assertEqual(nl['x_delta_linf_absolute_tol'], tolerance)
        self.assertEqual(nl['grad_norm_tol'], 0)
        self.assertEqual(nl['first_grad_norm_tol'], 0)
        self.assertEqual(nl['advanced']['f_delta'], 0)
        self.assertEqual(nl['advanced']['derivative_along_delta_x_tol'], 0)
        self.assertFalse(nl['allow_out_of_iterations'])

    def test_legacy_schema_preserves_absolute_tolerance(self):
        self.env['configure_newton_stopping'](self.config, 1e-3, 0.02, 2.7)
        class FakeSolver:
            def set_settings(self, settings): self.settings = json.loads(settings)
            def set_max_threads(self, threads): self.threads = threads
            def set_log_level(self, level): pass
            def load_mesh_from_settings(self): pass
        self.env['pf'] = types.SimpleNamespace(Solver=FakeSolver)
        solver = self.env['init_solver'](self.config, 0)
        nl = solver.settings['solver']['nonlinear']
        self.assertEqual(nl['grad_norm'], 0)
        self.assertEqual(nl['x_delta'], 0)
        self.assertAlmostEqual(nl['x_delta_linf_absolute_tol'], 5.4e-5)

    def test_modern_linf_leaves_bbox_scaling_to_native(self):
        self.env['_legacy_polysolve_schema'] = False
        tolerance = self.env['configure_newton_stopping'](self.config, 1e-3, 0.02, 2.7)
        nl = self.config['solver']['nonlinear']
        self.assertEqual(nl['norm_type'], 'Linf')
        self.assertAlmostEqual(nl['x_delta_tol'] * 2.7, tolerance)
        self.assertNotIn('x_delta_linf_absolute_tol', nl)
        self.assertEqual(nl['rel_grad_norm_tol'], 0)
        self.assertEqual(nl['rel_x_delta_tol'], 0)
        self.assertEqual(nl['newton_decrement_tol'], 0)
        self.assertEqual(nl['advanced']['f_delta_tol'], 0)

    def test_legacy_default_preserved(self):
        self.env['_newton_stopping'] = 'gradient'
        self.assertEqual(self.env['configure_newton_stopping'](self.config, 1e-3, 0.02, 2.7), 1e-3)
        self.assertNotIn('x_delta_linf_absolute_tol', self.config['solver']['nonlinear'])

    def test_invalid_bbox_rejected(self):
        for bbox in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.env['configure_newton_stopping'](self.config, 1e-3, 0.02, bbox)


if __name__ == '__main__':
    unittest.main()

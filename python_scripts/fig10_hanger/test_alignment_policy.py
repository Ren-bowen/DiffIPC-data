"""CPU regression checks for paired Fig.10 policies; no native solver required."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
import shlex
import subprocess
from unittest.mock import Mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UNIFIED = HERE.parents[2] / 'Unified_GIPC' / 'python_examples' / 'diff_sim'


def functions(path, names, **env):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names), (path, names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    env.update(np=np, torch=torch, Path=Path)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env


class AlignmentPolicyTests(unittest.TestCase):
    def test_design_selection_does_not_mask_coupled_state_derivatives(self):
        # A selected rest DOF moves a non-design state DOF. Its loss seed must
        # reach the adjoint even though the latter rest DOF cannot be updated.
        points = np.arange(12, dtype=np.float64).reshape(4, 3) / 10
        coupling = np.eye(12) * .2
        coupling[6, 0] = 2.
        current = points.copy()
        rhs = None

        def set_vertices(value):
            nonlocal current
            current = value.copy()

        def solve_adjoint(value):
            nonlocal rhs
            rhs = value.numpy().copy()

        mesh = SimpleNamespace(set_vertices=set_vertices)
        solver = SimpleNamespace(mesh=lambda: mesh, set_cache_level=lambda level: None,
            solve=lambda level: None, solve_adjoint=solve_adjoint,
            get_solutions=lambda: np.column_stack((np.zeros(12), coupling @ current.ravel())))
        pf = SimpleNamespace(CacheLevel=SimpleNamespace(Derivatives=1),
            shape_derivative=lambda solver: (coupling.T @ rhs[:, -1]).reshape(4, 3))
        env = functions(HERE / 'optimize.py', ['Simulate', 'completed_rollout_cache'],
                        pf=pf, time=time, _single_pass_profile=False)
        x = torch.tensor(points, requires_grad=True)
        design = torch.tensor([True, False, False, False])
        u = env['Simulate'].apply(solver, x, design, 1).reshape_as(x)
        loss = ((x + u) ** 2).sum()
        loss.backward()
        expected_seed = 2 * (points.ravel() + coupling @ points.ravel())
        np.testing.assert_allclose(rhs[:, -1], expected_seed)
        h = 1e-6

        def objective(value):
            return np.sum((value.ravel() + coupling @ value.ravel()) ** 2)

        plus = points.copy(); plus[0, 0] += h
        minus = points.copy(); minus[0, 0] -= h
        fd = (objective(plus) - objective(minus)) / (2 * h)
        self.assertAlmostEqual(float(x.grad[0, 0]), fd, places=7)

    def test_backward_does_not_modify_upstream_gradient(self):
        solver = SimpleNamespace(mesh=lambda: SimpleNamespace(set_vertices=lambda v: None),
            set_cache_level=lambda level: None, solve=lambda level: None,
            get_solutions=lambda: np.zeros((12, 2)), solve_adjoint=Mock())
        pf = SimpleNamespace(CacheLevel=SimpleNamespace(Derivatives=1),
                             shape_derivative=lambda solver: np.zeros((4, 3)))
        env = functions(HERE / 'optimize.py', ['Simulate', 'completed_rollout_cache'],
                        pf=pf, time=time, _single_pass_profile=False)
        x = torch.zeros((4, 3), dtype=torch.float64, requires_grad=True)
        output = env['Simulate'].apply(solver, x, torch.tensor([True, False, False, False]), 1)
        upstream = torch.arange(12, dtype=torch.float64)
        reference = upstream.clone()
        output.backward(upstream)
        torch.testing.assert_close(upstream, reference)

    def test_remesh_thresholds_ignore_relative_health_and_recent_attempt(self):
        attempts = Mock(return_value=None)
        env = functions(UNIFIED / 'fig10_hanger_optimization.py', ['_remesh_if_needed'],
                        remesh_with_ftetwild=attempts, tet_quality_stats=lambda *a: {},
                        _surface_regularization_value=lambda *a: 0., hanger_core_cap_metrics=lambda *a: {})
        context = SimpleNamespace(full_rest=lambda x: x, tets=[], faces=[], core_cap_ids=[])
        for q, scale, expected in [(0.003, .251, False), (.003, .25, True), (.002999, 1., True), (.004, 0., True)]:
            # A recent attempt and arbitrarily poor relative metrics cannot alter the decision.
            result = env['_remesh_if_needed'](context, np.zeros((4, 3)), Path('/tmp'), 0,
                enabled=True, iteration=11, last_attempt_iteration=10,
                mesh_health={'tet_quality_min': q, 'step_scale': scale, 'tet_relative_health_p01': 0.},
                quality_threshold=.003, step_scale_threshold=.25, max_remesh=100,
                remesh_retries=3, ftetwild_bin=None, ftetwild_opts=None)
            self.assertEqual(result[-1], expected)
            from feasibility_backtracking import needs_remesh
            self.assertEqual(needs_remesh(q, .003, scale), expected)
        self.assertEqual(attempts.call_count, 3)

    def test_surface_selection_and_cross_boundary_laplacian_match(self):
        env = functions(HERE / 'optimize.py', ['gipc_case13_design_mask', 'gipc_case13_design_band_mask',
            'gipc_case13_grad_shield_z_mask', 'hanger_arm_loaded_mask', 'hanger_arm_axis_planes',
            'gipc_case13_laplacian_surface',
            '_undirected_edges_from_triangles', 'laplacian_loss_torch'])
        ue = functions(
            UNIFIED / 'fig10_hanger_optimization.py',
            ['select_hanger_vertex_sets', 'hanger_arm_loaded_mask', 'hanger_arm_axis_planes',
             '_legacy_z_cut_plane', '_is_legacy_z_cut'],
            Y_LOWER=3.,
            Y_UPPER=5.,
            DESIGN_OUTER_HALF_WIDTH=1.,
            DESIGN_CORE_HALF_WIDTH=.25,
            ARM_Z_INNER=.25,
            ARM_Z_OUTER=.90,
        )
        ul = functions(UNIFIED / 'fig10_hanger_loss.py', ['_unique_edges', '_legacy_uniform_laplacian_loss'])
        # Vertex 3 is interior; triangle 0 crosses the strict y-band boundary.
        # This tiny stencil has no complete hanger arms, so isolate the band
        # and surface selection from the separately tested arm-plane fitting.
        unloaded = lambda points, require_fitted=False: np.zeros(len(points), dtype=bool)
        env['hanger_arm_loaded_mask'] = unloaded
        ue['hanger_arm_loaded_mask'] = unloaded
        v = np.array([[0, 3.2, 0], [1, 3.8, 0], [0, 2.8, 0], [.2, 3.5, .1], [0, 4., 2.]])
        faces = np.array([[0, 1, 2], [0, 2, 4], [1, 4, 2], [0, 4, 1]])
        sets = ue['select_hanger_vertex_sets'](v, faces, laplacian_mode='legacy_uniform', design_surface_only=True)
        selected = env['gipc_case13_design_mask'](v, (v[:, 1]<3)|(v[:, 1]>5), faces)
        np.testing.assert_array_equal(np.flatnonzero(selected), sets['selected_ids'])
        self.assertFalse(selected[3])
        f, ids = env['gipc_case13_laplacian_surface'](v, faces, len(v))
        np.testing.assert_array_equal(ids, sets['band_ids'])
        # Freeze ids, then move a selected vertex across y=3 during a trial.
        trial = v.copy(); trial[0, 1] = 2.9
        x = torch.tensor(trial, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a = env['laplacian_loss_torch'](x, torch.tensor(f), torch.tensor(ids))
        b = ul['_legacy_uniform_laplacian_loss'](y, torch.tensor(faces), torch.tensor(sets['band_ids']))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(torch.autograd.grad(a, x)[0], torch.autograd.grad(b, y)[0])

    def test_trial_stress_uses_frozen_region(self):
        captured = []
        env = functions(HERE / 'optimize.py', ['eval_stress_norm_loss', 'gipc_case13_loss_tet_mask', 'completed_rollout_cache'],
                        pf=SimpleNamespace(CacheLevel=SimpleNamespace(Derivatives=1)),
                        stable_nh_stress_norm_loss=lambda d, r, t, p, mask, stress_weight=1.0:
                        captured.append(mask.numpy().copy()) or torch.tensor(1.))
        base = np.array([[0,3.1,0], [1,3.1,0], [0,3.2,0], [0,3.1,1.]])
        tets = np.array([[0,1,2,3]])
        fixed = env['gipc_case13_loss_tet_mask'](base, tets, 4)
        trial = base.copy(); trial[:,1] -= 1.
        mesh = SimpleNamespace(set_vertices=lambda v: None, elements=lambda: tets)
        solver = SimpleNamespace(mesh=lambda: mesh, set_cache_level=lambda v: None,
                                 solve=lambda: None, get_solutions=lambda: np.zeros((12,21)))
        env['eval_stress_norm_loss'](solver, trial, 2, 4, strict_solve=True, tet_mask=fixed)
        self.assertTrue(captured[0][0])
        self.assertFalse(env['gipc_case13_loss_tet_mask'](trial, tets, 4)[0])

    def test_rollout_failure_and_incomplete_cache_are_rejected(self):
        env = functions(HERE / 'optimize.py', ['Simulate', 'completed_rollout_cache', 'eval_stress_norm_loss'],
                        pf=SimpleNamespace(CacheLevel=SimpleNamespace(Derivatives=1)))
        vertices = torch.zeros((4, 3), dtype=torch.float64)
        solver = SimpleNamespace(mesh=lambda: SimpleNamespace(set_vertices=lambda v: None),
                                 set_cache_level=lambda v: None, solve=Mock(),
                                 get_solutions=Mock(return_value=np.zeros((12, 21))))
        def forward():
            return env['Simulate'].apply(solver, vertices, torch.ones(4, dtype=torch.bool), 20)
        # Even a full/stale cache must not hide a solve failure.
        solver.solve.side_effect = RuntimeError('Newton failed')
        for evaluate in (forward, lambda: env['eval_stress_norm_loss'](solver, vertices.numpy(), 2, 4)):
            with self.assertRaisesRegex(RuntimeError, 'Newton failed'):
                evaluate()
        solver.get_solutions.assert_not_called()
        solver.solve.side_effect = None
        for cache in (np.zeros((12, 12)), np.zeros((12, 1)), np.zeros((12, 0)),
                      np.zeros(12), np.zeros((9, 21)), np.zeros((12, 22))):
            solver.get_solutions.return_value = cache
            for evaluate in (forward, lambda: env['eval_stress_norm_loss'](solver, vertices.numpy(), 2, 4)):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete or invalid'):
                    evaluate()
        cache = np.zeros((12, 21)); cache[0, 3] = np.nan
        solver.get_solutions.return_value = cache
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            forward()
        cache[0, 3] = 0.; cache[:, -1] = np.arange(12)
        torch.testing.assert_close(forward(), torch.arange(12, dtype=torch.float64))
        # A non-default frame budget must be honored as well.
        solver.get_solutions.return_value = np.zeros((12, 8))
        result = env['Simulate'].apply(solver, vertices, torch.ones(4, dtype=torch.bool), 7)
        self.assertEqual(result.shape, (12,))
        env.update(time=time, _single_pass_profile=False)
        env['pf'].shape_derivative = lambda solver: np.ones((4, 3))
        solver.solve_adjoint = Mock()
        vertices.requires_grad_()
        env['Simulate'].apply(solver, vertices, torch.ones(4, dtype=torch.bool), 7).sum().backward()
        torch.testing.assert_close(vertices.grad, torch.ones_like(vertices))
        self.assertEqual(solver.solve_adjoint.call_args.args[0].shape, (12, 8))

    def test_accepted_forward_reuses_native_cache_once_and_preserves_backward(self):
        pf = SimpleNamespace(CacheLevel=SimpleNamespace(Derivatives=1),
                             shape_derivative=lambda solver: np.full((4, 3), 2.))
        env = functions(HERE / 'optimize.py', ['Simulate', 'AcceptedForward', 'completed_rollout_cache'],
                        pf=pf, time=time, _single_pass_profile=False)
        points = np.zeros((4, 3)); tets = np.array([[0, 1, 2, 3]])
        mesh = SimpleNamespace(vertices=lambda: points.copy(), elements=lambda: tets.copy(), set_vertices=Mock())
        solver = SimpleNamespace(mesh=lambda: mesh, solve=Mock(), set_cache_level=Mock(),
                                 get_solutions=lambda: np.zeros((12, 21)), solve_adjoint=Mock())
        # The accepted trial already solved. Next iteration must not reset or solve.
        token = env['AcceptedForward'](solver, points, 20)
        x = torch.tensor(points, requires_grad=True)
        env['Simulate'].apply(solver, x, torch.ones(4, dtype=torch.bool), 20, token).sum().backward()
        solver.solve.assert_not_called(); mesh.set_vertices.assert_not_called()
        solver.set_cache_level.assert_not_called(); solver.solve_adjoint.assert_called_once()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2.))
        # Consumed tokens cannot silently reuse a cache again.
        env['Simulate'].apply(solver, x, torch.ones(4, dtype=torch.bool), 20, token)
        self.assertEqual(solver.solve.call_count, 1)
        for mismatch in ('parameters', 'frames', 'topology', 'solver'):
            token = env['AcceptedForward'](solver, points, 20)
            candidate = points.copy(); frames = 20; other = solver
            if mismatch == 'parameters': candidate[0, 0] = 1.
            if mismatch == 'frames': frames = 19
            if mismatch == 'topology': tets[0, 0] = 2
            if mismatch == 'solver': other = SimpleNamespace()
            self.assertIsNone(token.take(other, candidate, frames))
            tets[0, 0] = 0

    def test_paired_ftetwild_resolution_and_envelope(self):
        # Execute command builders with a fake executable failure (no expensive meshing).
        paths = [UNIFIED/'2_fig10_hanger_rest_shape'/'fig10_remesh.py', HERE.parent/'src'/'tet_remesh.py']
        commands = []
        fake = SimpleNamespace(STDOUT=subprocess.STDOUT, run=lambda cmd, **kw: commands.append(cmd) or SimpleNamespace(returncode=1), TimeoutExpired=subprocess.TimeoutExpired)
        points = np.array([[0.,0,0], [1,0,0], [0,1,0], [0,0,1]])
        tets = np.array([[0,1,2,3]])
        faces = np.array([[0,1,2]])
        with tempfile.TemporaryDirectory() as tmp:
            for path in paths:
                tree = ast.parse(path.read_text()); constants = {}
                for n in tree.body:
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('DEFAULT_FTETWILD_'):
                        constants[n.targets[0].id] = ast.literal_eval(n.value)
                self.assertAlmostEqual(constants['DEFAULT_FTETWILD_LA'], .035388307704147845)
                name = 'remesh_with_ftetwild' if path == paths[0] else 'run_remesh'
                env = functions(path, [name], **constants, shlex=shlex, subprocess=fake,
                    _resolve_binary=lambda *a: 'fake', _boundary_faces=lambda *a: faces,
                    _outward_boundary_faces=lambda *a: faces,
                    _write_surface_obj=lambda *a: None, write_surface_obj=lambda *a: None)
                if name == 'remesh_with_ftetwild':
                    env[name](points, tets, Path(tmp), 1, retries=2)
                else:
                    for attempt in (1,2):
                        env[name](tmp, 1, attempt, points, tets, ftetwild_epsilon=.001)
        for a,b in zip(commands[:2], commands[2:]):
            for option in ('--la', '--epsr', '--max-threads'):
                self.assertEqual(a[a.index(option)+1], b[b.index(option)+1])
            self.assertIn('--no-binary', a); self.assertIn('--no-binary', b)


if __name__ == '__main__':
    unittest.main()

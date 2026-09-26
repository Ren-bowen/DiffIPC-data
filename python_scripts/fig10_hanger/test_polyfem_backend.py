"""Backend selection is fail-closed, without importing a native library."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import polyfem_backend as backend


class BackendTests(unittest.TestCase):
    def test_missing_build_does_not_import_installed_pinned(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(backend, 'REN_BOWEN_PACKAGE_ROOT', Path(tmp)), \
                patch.object(backend.importlib, 'import_module') as importer:
            with self.assertRaisesRegex(RuntimeError, 'No pinned fallback'):
                backend.load_polyfem('ren-bowen')
            importer.assert_not_called()

    def test_ren_bowen_rejects_pinned_provenance(self):
        pf = types.SimpleNamespace(__file__='/old/polyfempy/__init__.py',
                                   build_info={'source_dir': '/home/bowen/polyfem-pinned'})
        with patch.object(Path, 'glob', return_value=iter([Path('module.so')])), \
                patch.object(backend.importlib, 'import_module', return_value=pf), \
                patch.object(backend.sys, 'path', []):
            with self.assertRaisesRegex(RuntimeError, 'Wrong PolyFEM core'):
                backend.load_polyfem('ren-bowen')

    def test_pinned_rejects_modern_import(self):
        pf = types.SimpleNamespace(build_info={'api': 'split-state-fig10'})
        with patch.object(backend.importlib, 'import_module', return_value=pf):
            with self.assertRaisesRegex(RuntimeError, 'Pinned requested'):
                backend.load_polyfem('pinned')

    def test_ren_bowen_accepts_matching_package_and_core(self):
        pf = types.SimpleNamespace(
            __file__=str(backend.REN_BOWEN_PACKAGE_ROOT / 'polyfempy/__init__.py'),
            build_info={'source_dir': str(backend.REN_BOWEN_CORE)})
        with patch.object(Path, 'glob', return_value=iter([Path('module.so')])), \
                patch.object(backend.importlib, 'import_module', return_value=pf), \
                patch.object(backend.sys, 'path', []):
            self.assertIs(backend.load_polyfem('ren-bowen'), pf)


if __name__ == '__main__':
    unittest.main()

"""Select the Fig10 binding explicitly; never fall back to a different core."""
import importlib
from pathlib import Path
import sys

REN_BOWEN_CORE = Path('/home/bowen/polyfem')
REN_BOWEN_PACKAGE_ROOT = Path('/home/bowen/polyfem-python/build-renbowen')


def load_polyfem(backend):
    if backend == 'ren-bowen':
        package = REN_BOWEN_PACKAGE_ROOT / 'polyfempy'
        if not list(package.glob('polyfempy*.so')):
            raise RuntimeError(f'Ren-bowen binding has not been built in {package}; '
                               'see REN_BOWEN_BINDING.md. No pinned fallback is used.')
        sys.path.insert(0, str(REN_BOWEN_PACKAGE_ROOT))
    elif backend != 'pinned':
        raise ValueError(f'Unknown PolyFEM backend: {backend}')
    pf = importlib.import_module('polyfempy')
    info = getattr(pf, 'build_info', {})
    if backend == 'ren-bowen':
        if Path(info.get('source_dir', '')).resolve() != REN_BOWEN_CORE.resolve():
            raise RuntimeError(f'Wrong PolyFEM core loaded: {info}, module={pf.__file__}')
        if not Path(pf.__file__).resolve().is_relative_to(REN_BOWEN_PACKAGE_ROOT.resolve()):
            raise RuntimeError(f'Wrong PolyFEM package loaded: {pf.__file__}')
    elif info.get('api') == 'split-state-fig10':
        raise RuntimeError('Pinned requested, but Ren-bowen binding found on PYTHONPATH')
    return pf

"""Run aligned Fig.1 optimization with target excluded from optimization timing."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()
    driver = Path('/home/bowen/Unified_GIPC/agent_check/diff_sim/fig1_default_driver_20260915/target_first')
    if not driver.is_file():
        raise FileNotFoundError(driver)
    out = (args.out_dir or here / ('run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for state in ('run', 'target'):
        data = json.loads((here / (state + '.json')).read_text())
        for geometry in data['geometry']:
            geometry['mesh'] = str((here / geometry['mesh']).resolve())
        data['output']['directory'] = str(out / ('state_' + state))
        data['output']['log']['level'] = 'trace'
        data['output']['advanced']['save_time_sequence'] = False
        data['output']['paraview'].update(volume=False, surface=False, skip_frame=1000000000)
        (out / (state + '.json')).write_text(json.dumps(data, indent=2))
    opt = json.loads((here / 'opt.json').read_text())
    opt['states'] = [{'path': str(out / (state + '.json'))} for state in ('run', 'target')]
    (out / 'opt.json').write_text(json.dumps(opt, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='32', MKL_NUM_THREADS='32', OPENBLAS_NUM_THREADS='32', MKL_DYNAMIC='FALSE')
    env['LD_LIBRARY_PATH'] = '/home/bowen/miniconda3/envs/env_isaaclab/lib:' + env.get('LD_LIBRARY_PATH', '')
    print(f'Output: {out}', flush=True)
    with (out / 'run.log').open('w') as log:
        result = subprocess.run([str(driver), str(out / 'opt.json'), str(out / 'driver_result.json')], cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

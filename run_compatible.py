#!/usr/bin/env python3
"""Run an explicit command with the verified per-container Hygon topology adapter."""
import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from compat_topology import IMAGE_ID, build_overlays

ROOT = Path(__file__).resolve().parent


def inspect_container(target):
    p = subprocess.run(['docker', 'inspect', target], capture_output=True, text=True, timeout=30)
    if p.returncode:
        if 'No such' in p.stderr:
            return None
        raise RuntimeError('docker inspect failed: ' + p.stderr)
    return json.loads(p.stdout)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not re.fullmatch('[a-z][a-z0-9_-]{0,45}', args.case) or not 1 <= args.timeout <= 7200:
        parser.error('invalid case or timeout')
    if not command or not command[0].startswith('/'):
        parser.error('provide an absolute executable after --')
    work = ROOT / 'work'
    if not work.is_dir() or work.is_symlink():
        raise ValueError('private work directory required')
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE_ID], text=True, timeout=30))[0]
    if image['Id'] != IMAGE_ID or image['Config'].get('Volumes') or image['Config'].get('Healthcheck'):
        raise ValueError('unverified image identity/config')
    inherited = dict(x.split('=', 1) for x in image['Config'].get('Env', []))
    if any(k.startswith(('LD_', 'HSA_', 'HIP_', 'ROCR_')) for k in inherited):
        raise ValueError('unexpected inherited runtime override')
    run_id = uuid.uuid4().hex
    run = ROOT / 'runs' / (args.case + '-' + run_id[:12])
    run.mkdir(parents=True, exist_ok=False)
    overlays = build_overlays(run / 'topology')
    (run / 'topology.json').write_text(json.dumps(overlays, indent=2) + '\n')
    name = 'ml-hygon-' + run_id
    create = ['docker', 'create', '--name', name,
              '--label', 'ml.hygon-z100l.adaptation=1', '--label', 'ml.hygon-z100l.run=' + run_id,
              '--read-only', '--network', 'none', '--cap-drop', 'ALL',
              '--security-opt', 'no-new-privileges', '--no-healthcheck',
              '--pids-limit', '256', '--memory', '16g', '--cpus', '8',
              '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g', '--ulimit', 'core=0',
              '--ulimit', 'memlock=-1:-1', '--workdir', '/work',
              '--mount', 'type=bind,source=' + str(work) + ',target=/work',
              '--device', '/dev/kfd', '--device', '/dev/mkfd', '--device', '/dev/dri']
    for row in overlays:
        create += ['--mount', 'type=bind,source=' + row['source'] + ',target=' + row['target'] + ',readonly']
    create += ['--entrypoint', command[0], IMAGE_ID] + command[1:]
    (run / 'command.json').write_text(json.dumps(create, indent=2) + '\n')
    container_id = None
    result = {'case': args.case, 'run_id': run_id, 'image_id': IMAGE_ID,
              'command': command, 'scope': 'container topology adapter only; no LD_PRELOAD, ISA override env, host HIP/HSA libraries or driver replacement'}
    start = time.monotonic()
    try:
        p = subprocess.run(create, capture_output=True, text=True, timeout=600)
        (run / 'create.json').write_text(json.dumps({'exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}, indent=2) + '\n')
        p.check_returncode()
        container_id = p.stdout.strip()
        obj = inspect_container(container_id)
        if obj is None or obj['Image'] != IMAGE_ID or obj['Config']['Labels'].get('ml.hygon-z100l.run') != run_id:
            raise ValueError('created container identity')
        (run / 'container.json').write_text(json.dumps(obj, indent=2) + '\n')
        with (run / 'stdout.log').open('x') as log:
            p = subprocess.run(['docker', 'start', '--attach', container_id], stdout=log,
                               stderr=subprocess.STDOUT, timeout=args.timeout)
        state = inspect_container(container_id)['State']
        result.update(docker_start_exit=p.returncode, state=state)
        code = state['ExitCode']
        if state['Running'] or state['OOMKilled'] or state['Error'] or p.returncode != code:
            raise RuntimeError('container execution state mismatch')
    except subprocess.TimeoutExpired:
        result['timeout'] = True
        raise
    except BaseException as exc:
        result['error_type'] = type(exc).__name__
        raise
    finally:
        result['seconds'] = time.monotonic() - start
        try:
            obj = inspect_container(container_id or name)
            if obj is None:
                result['cleanup'] = {'not_present_when_checked': True,
                                     'note': 'if create timed out, later daemon completion requires owned-run reconciliation'}
            else:
                if obj['Image'] != IMAGE_ID or obj['Config']['Labels'].get('ml.hygon-z100l.run') != run_id:
                    raise ValueError('refuse unowned container removal')
                p = subprocess.run(['docker', 'rm', '--force', '--volumes', obj['Id']],
                                   capture_output=True, text=True, timeout=60)
                result['cleanup'] = {'container_id': obj['Id'], 'remove_exit': p.returncode}
                p.check_returncode()
        finally:
            (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'case': args.case, 'evidence': str(run), 'result': result}), flush=True)
    print((run / 'stdout.log').read_text()[-16000:], flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())

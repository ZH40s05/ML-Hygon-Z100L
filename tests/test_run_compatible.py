import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import run_compatible as runner
from compat_topology import IMAGE_ID


class LauncherTests(unittest.TestCase):
    def execute(self, exit_code=0, oom=False, foreign=False, image_env=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'work').mkdir()
            image = {'Id': IMAGE_ID, 'Config': {'Env': image_env or ['ROCM_PATH=/opt/rocm']}}
            state = {'ExitCode': exit_code, 'Running': False, 'OOMKilled': oom, 'Error': ''}
            calls = []
            identity = {}

            def run(argv, **kwargs):
                calls.append(argv)
                if argv[:2] == ['docker', 'create']:
                    label_index = argv.index('ml.hygon-z100l.run=' + argv[argv.index('--name') + 1].removeprefix('ml-hygon-'))
                    identity.update(Image=IMAGE_ID, Id='owned-container',
                                    Config={'Labels': {'ml.hygon-z100l.run': argv[label_index].split('=', 1)[1]}},
                                    State=state)
                    if foreign: identity['Image'] = 'sha256:foreign'
                    return subprocess.CompletedProcess(argv, 0, 'owned-container\n', '')
                if argv[:2] == ['docker', 'start']:
                    return subprocess.CompletedProcess(argv, exit_code)
                if argv[:2] == ['docker', 'rm']:
                    return subprocess.CompletedProcess(argv, 0, 'owned-container\n', '')
                self.fail('unexpected command: ' + repr(argv))

            overlays = [{'source': str(root / 'overlay' / str(i)),
                         'target': '/sys/devices/virtual/kfd/kfd/topology/nodes/' + str(i) + '/properties'}
                        for i in range(1, 9)]
            with patch.object(runner, 'ROOT', root), \
                 patch.object(sys, 'argv', ['run_compatible.py', '--case', 'unit-case', '--', '/work/probe']), \
                 patch.object(runner, 'build_overlays', return_value=overlays), \
                 patch.object(runner.subprocess, 'check_output', return_value=json.dumps([image])), \
                 patch.object(runner.subprocess, 'run', side_effect=run), \
                 patch.object(runner, 'inspect_container', side_effect=lambda _: copy.deepcopy(identity)), \
                 redirect_stdout(io.StringIO()):
                try:
                    code = runner.main()
                    error = None
                except ValueError as exc:
                    code, error = None, str(exc)
                except RuntimeError as exc:
                    code, error = None, str(exc)
            results = list((root / 'runs').glob('*/result.json')) if (root / 'runs').exists() else []
            result = json.loads(results[0].read_text()) if results else None
            return code, error, calls, result

    def test_success_uses_only_readonly_topology_and_private_work_mounts(self):
        code, error, calls, result = self.execute()
        self.assertEqual(code, 0)
        self.assertIsNone(error)
        create = next(c for c in calls if c[:2] == ['docker', 'create'])
        mounts = [create[i + 1] for i, v in enumerate(create[:-1]) if v == '--mount']
        self.assertEqual(len(mounts), 9)
        self.assertEqual(sum(m.endswith(',readonly') for m in mounts), 8)
        self.assertIn('--read-only', create)
        self.assertIn('no-new-privileges', create)
        self.assertNotIn('--privileged', create)
        self.assertNotIn('--env', create)
        self.assertEqual(result['cleanup']['remove_exit'], 0)
        self.assertEqual(calls[-1][:2], ['docker', 'rm'])

    def test_nonzero_workload_exit_propagates_after_owned_cleanup(self):
        code, error, calls, result = self.execute(exit_code=2)
        self.assertEqual(code, 2)
        self.assertIsNone(error)
        self.assertEqual(result['state']['ExitCode'], 2)
        self.assertEqual(result['cleanup']['remove_exit'], 0)

    def test_oom_is_failure_even_with_zero_reported_exit(self):
        code, error, calls, result = self.execute(oom=True)
        self.assertIsNone(code)
        self.assertEqual(error, 'container execution state mismatch')
        self.assertEqual(result['cleanup']['remove_exit'], 0)

    def test_foreign_container_never_removed(self):
        code, error, calls, result = self.execute(foreign=True)
        self.assertEqual(error, 'refuse unowned container removal')
        self.assertFalse(any(c[:2] == ['docker', 'rm'] for c in calls))

    def test_inherited_runtime_override_refused_before_container_creation(self):
        code, error, calls, result = self.execute(image_env=['HSA_OVERRIDE_GFX_VERSION=9.0.6'])
        self.assertEqual(error, 'unexpected inherited runtime override')
        self.assertEqual(calls, [])
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

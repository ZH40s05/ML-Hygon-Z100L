import json
import tempfile
import unittest
from pathlib import Path
from compat_topology import compatibility_properties, parse_properties, build_overlays, DRIVER_SRCVERSION

BASE = Path(__file__).resolve().parent / 'fixtures' / 'topology.json'
TOPOLOGY = json.loads(BASE.read_text())
GPU = next(row['properties'] for row in TOPOLOGY.values() if 'gfx_target_version 90106' in row.get('properties', ''))


class CompatibilityTests(unittest.TestCase):
    def test_exact_hardware_and_vendor_formula(self):
        text, info = compatibility_properties(GPU)
        props = parse_properties(text)
        self.assertEqual(props['gfx_target_version'], 90006)
        self.assertEqual(props['vendor_id'], 0x1d94)
        self.assertEqual(props['device_id'], 0x55b7)
        self.assertEqual((info['ctl_stack_size'], info['cwsr_size']), (0x5000, 0x1fff000))
        self.assertEqual(info['upstream_uncapped_cwsr_size'], 0x22f1000)
        original = parse_properties(GPU)
        self.assertEqual({k: v for k, v in props.items() if k not in ('gfx_target_version', 'cwsr_size', 'ctl_stack_size')},
                         {k: v for k, v in original.items() if k != 'gfx_target_version'})

    def test_non_hygon_rejected(self):
        with self.assertRaises(ValueError):
            compatibility_properties(GPU.replace('vendor_id 7572', 'vendor_id 4098'))

    def test_other_native_isa_rejected(self):
        with self.assertRaises(ValueError):
            compatibility_properties(GPU.replace('gfx_target_version 90106', 'gfx_target_version 90006'))

    def test_different_cu_profile_rejected(self):
        with self.assertRaises(ValueError):
            compatibility_properties(GPU.replace('simd_count 240', 'simd_count 256'))

    def test_conflicting_driver_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            compatibility_properties(GPU + 'cwsr_size 36638720\n')

    def test_matching_driver_dimensions_not_duplicated(self):
        text, _ = compatibility_properties(GPU + 'cwsr_size 33550336\nctl_stack_size 20480\n')
        self.assertEqual(text.count('cwsr_size '), 1)
        self.assertEqual(text.count('ctl_stack_size '), 1)

    def test_duplicate_property_rejected(self):
        with self.assertRaises(ValueError):
            compatibility_properties(GPU + 'gfx_target_version 90106\n')

    def test_malformed_property_rejected(self):
        with self.assertRaises(ValueError):
            compatibility_properties(GPU + 'missing_value\n')

    def make_topology(self, root, omit=None):
        for name, row in TOPOLOGY.items():
            if name == omit:
                continue
            node = root / name
            node.mkdir()
            for key, value in row.items():
                (node / key).write_text(value)

    def test_eight_readonly_overlays_preserve_host_properties(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'sysfs'; root.mkdir()
            self.make_topology(root)
            before = {p: p.read_bytes() for p in root.glob('*/properties')}
            rows = build_overlays(Path(temp) / 'overlay', root, '6.2.26', DRIVER_SRCVERSION)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r['gpu_id'] for r in rows}), 8)
            for row in rows:
                self.assertEqual(Path(row['source']).stat().st_mode & 0o222, 0)
                self.assertEqual(parse_properties(Path(row['source']).read_text())['cwsr_size'], 33550336)
            for p, b in before.items(): self.assertEqual(p.read_bytes(), b)

    def test_missing_gpu_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'sysfs'; root.mkdir()
            name = next(k for k,v in TOPOLOGY.items() if v.get('gpu_id','').strip() not in ('', '0'))
            self.make_topology(root, omit=name)
            with self.assertRaises(ValueError):
                build_overlays(Path(temp) / 'overlay', root, '6.2.26', DRIVER_SRCVERSION)

    def test_other_loaded_driver_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                build_overlays(Path(temp) / 'overlay', Path(temp), '6.2.26', 'DIFFERENT')

    def test_other_driver_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                build_overlays(Path(temp) / 'overlay', Path(temp), '6.4.2')


if __name__ == '__main__':
    unittest.main()

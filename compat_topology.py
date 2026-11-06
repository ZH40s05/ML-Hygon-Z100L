"""Narrow, fail-closed topology compatibility for this eight-card Hygon rig.

Only container-visible properties change. Host sysfs, libraries and driver stay intact.
CWSR allocation/header initialization still run in the unmodified community thunk.
"""
from pathlib import Path
import hashlib

IMAGE_ID = 'sha256:c5632b04e8874537ffb597c2037179fc62d2f7b7c8ad77adab342d3ef1109ef2'
TOPOLOGY_ROOT = Path('/sys/devices/virtual/kfd/kfd/topology/nodes')
DRIVER_SRCVERSION = 'B568DE62D6C1131D653C14F'


def parse_properties(text):
    props = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError('malformed KFD property line')
        key, value = fields
        if key in props:
            raise ValueError('duplicate KFD property: ' + key)
        props[key] = int(value, 0)
    return props


def align_up(value, alignment=4096):
    return (value + alignment - 1) // alignment * alignment


def compatibility_properties(text):
    """Match only verified Z100L/Hygon topology and derive the vendor size rule."""
    p = parse_properties(text)
    expected = {'vendor_id': 0x1d94, 'device_id': 0x55b7,
                'gfx_target_version': 90106, 'wave_front_size': 64,
                'simd_count': 240, 'simd_per_cu': 4,
                'lds_size_in_kb': 64, 'vgpr_size_per_cu': 524288,
                'sgpr_size_per_cu': 16384, 'dp4x_status': 0}
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError('unverified hardware property: ' + key)
    cu_count = p['simd_count'] // p['simd_per_cu']
    # Original hydcu6.2.26 kfd_chardev.c check_ctx_size_invalid, CHIP_ZIFANG.
    control = min(align_up(40 + cu_count * 40 * 8 + 8), 0x7000)
    wave_data = cu_count * (p['sgpr_size_per_cu'] + p['vgpr_size_per_cu']
                            + p['lds_size_in_kb'] * 1024 + 0x1000)
    uncapped = control + align_up(wave_data)
    cwsr = min(uncapped, 0x1fff000)
    if (control, cwsr, uncapped) != (0x5000, 0x1fff000, 0x22f1000):
        raise ValueError('unverified vendor CWSR dimensions')
    for key, value in [('cwsr_size', cwsr), ('ctl_stack_size', control)]:
        if key in p and p[key] != value:
            raise ValueError('conflicting driver-provided property: ' + key)
    additions = {'cwsr_size': cwsr, 'ctl_stack_size': control}
    # 90106 is the vendor native encoding, not the supported AMD target ID.
    # Explicitly report the already numerically verified gfx906 ISA to this container.
    lines = []
    for line in text.splitlines():
        key = line.split()[0]
        if key == 'gfx_target_version':
            lines.append('gfx_target_version 90006')
        elif key in additions:
            lines.append(key + ' ' + str(additions.pop(key)))
        else:
            lines.append(line)
    lines.extend(key + ' ' + str(value) for key, value in additions.items())
    compat = '\n'.join(lines) + '\n'
    info = {'physical_vendor_id': p['vendor_id'], 'physical_device_id': p['device_id'],
            'native_gfx_target_version': 90106, 'compat_gfx_target_version': 90006,
            'cu_count': cu_count, 'ctl_stack_size': control, 'cwsr_size': cwsr,
            'upstream_uncapped_cwsr_size': uncapped,
            'original_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'compat_sha256': hashlib.sha256(compat.encode()).hexdigest()}
    return compat, info


def build_overlays(output, topology_root=TOPOLOGY_ROOT, driver_version=None, driver_srcversion=None):
    version = driver_version if driver_version is not None else Path('/sys/module/hydcu/version').read_text().strip()
    if version != '6.2.26':
        raise ValueError('unverified hydcu version')
    srcversion = driver_srcversion if driver_srcversion is not None else Path('/sys/module/hydcu/srcversion').read_text().strip()
    if srcversion != DRIVER_SRCVERSION:
        raise ValueError('unverified loaded hydcu source identity')
    output = Path(output)
    output.mkdir(exist_ok=False)
    records = []
    gpu_nodes = []
    for node in sorted(Path(topology_root).iterdir(), key=lambda p: int(p.name)):
        gpu_id_path = node / 'gpu_id'
        if not gpu_id_path.is_file():
            continue
        gpu_id = int(gpu_id_path.read_text().strip())
        if gpu_id == 0:
            continue
        gpu_nodes.append((node, gpu_id))
    if len(gpu_nodes) != 8 or len(set(g for _, g in gpu_nodes)) != 8:
        raise ValueError('exactly eight unique GPUs required')
    render_minors = set()
    for node, gpu_id in gpu_nodes:
        target = node / 'properties'
        original = target.read_text()
        compat, info = compatibility_properties(original)
        minor = parse_properties(original)['drm_render_minor']
        if minor in render_minors:
            raise ValueError('duplicate render node')
        render_minors.add(minor)
        dest = output / (node.name + '.properties')
        dest.write_text(compat)
        dest.chmod(0o444)
        records.append(dict(info, node=node.name, gpu_id=gpu_id,
                            drm_render_minor=minor, source=str(dest.resolve()),
                            target=str(target), driver_version=version,
                            driver_srcversion=srcversion))
    return records

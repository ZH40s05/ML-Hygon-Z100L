# Compatibility mechanism

The observed failure chain was HIP kernel loading → HSA executable freeze → internal utility queue → KFD CREATE_QUEUE → EFAULT. The driver log showed required control/CWSR sizes of 0x5000/0x1fff000 and supplied sizes of 0x5000/0x22f1000. The Hygon driver checks these dimensions before creating a queue.

For the Z100L profile with dp4x disabled:

~~~text
CU = SIMD_count / SIMD_per_CU = 240 / 4 = 60
control = min(align4096(40 + CU * 40 * 8 + 8), 0x7000) = 20480
wave_data = CU * (SGPR + VGPR + LDS + 4096)
          = 60 * (16384 + 524288 + 65536 + 4096) = 36618240
upstream_total = control + align4096(wave_data) = 36638720
vendor_total = min(upstream_total, 0x1fff000) = 33550336
~~~

The adapter supplies both dimensions through KFD topology properties. The community runtime reads them before allocating and initializing the context-save area, keeping the allocation, header and queue dimensions consistent.

It also maps the native target 90106 to 90006/gfx906 within the container. PCI vendor and device fields remain 1d94:55b7.

Before generating overlays, the launcher checks the eight GPU IDs and render nodes, hydcu version/source identity, PCI IDs, CU profile and existing properties. The README lists the supported environment.

## Diagnostic controls

| Control | Outcome |
|---|---|
| gfx906 ISA mapping alone | Enumerates 8; first kernel fails 209 |
| ioctl request size 96→88 | CREATE_QUEUE still EFAULT |
| HSA_USE_SVM=0 | First kernel still fails |
| HSA_USE_SVM=0 + HSA_USERPTR_FOR_PAGED_MEM=0 | First kernel still fails |
| Correct CWSR/ctl properties with ISA mapping | Single/eight/56-way tests pass |

The initial CWSR control used an ioctl logger and explicit ISA override. Final hardware validation used the topology overlay with the community image's HIP/HSA libraries.

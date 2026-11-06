# Compatibility mechanism

The failure chain was observed as HIP kernel loading → HSA executable freeze → lazy internal utility queue → KFD CREATE_QUEUE → EFAULT. The vendor driver logs separately printed calculated control/CWSR sizes 0x5000/0x1fff000 and supplied sizes 0x5000/0x22f1000. The driver source from the original local installation package contained a strict equality check before queue creation. That vendor source is not redistributed here.

For the verified non-dp4x Z100L profile:

```text
CU = SIMD_count / SIMD_per_CU = 240 / 4 = 60
control = min(align4096(40 + CU * 40 * 8 + 8), 0x7000) = 20480
wave_data = CU * (SGPR + VGPR + LDS + 4096)
          = 60 * (16384 + 524288 + 65536 + 4096) = 36618240
upstream_total = control + align4096(wave_data) = 36638720
vendor_total = min(upstream_total, 0x1fff000) = 33550336
```

The overlay supplies both dimensions through existing topology properties. The unmodified runtime reads them before allocating and initializing the context-save area. This keeps allocation, header initialization and the submitted dimensions consistent. Merely truncating the ioctl field after allocation was not used as the fix.

The topology adapter also maps native vendor target90106 to verified AMD target90006/gfx906 for this container. PCI vendor and device identity remain Hygon1d94:55b7; the generic AMD Radeon Graphics runtime name is not a claim that the physical devices are AMD products.

The current adapter rejects a different hydcu version/source identity, nonmatching vendor/device/ISA/CU fields, duplicate properties, malformed lines, conflicting dimensions, any GPU count other than eight, duplicate GPU IDs or render minors. It intentionally makes no compatibility claim for different firmware, hardware or drivers.

Negative controls in the original experiment:

| Control | Outcome |
|---|---|
| gfx906 ISA mapping alone | Enumerates8; first kernel fails209 |
| ioctl request size96→88 | CREATE_QUEUE still EFAULT |
| HSA_USE_SVM=0 | First kernel still fails |
| HSA_USE_SVM=0 + HSA_USERPTR_FOR_PAGED_MEM=0 | First kernel still fails |
| Correct CWSR/ctl properties with ISA mapping | Single/eight/56-way correctness passes |

Initial CWSR control used an instrumentation-only ioctl logger and explicit ISA override. Final hardware validation used topology overlays alone with original community HIP/HSA libraries. No cache invalidation was skipped, no driver check was disabled, and no host global library state was changed.

The export keeps the numerical workloads and topology adapter unchanged. Only checkout-relative storage and neutral container labels/names were introduced in the public launcher. Unit tests cover this launcher; a new hardware run of that portable export is not claimed.

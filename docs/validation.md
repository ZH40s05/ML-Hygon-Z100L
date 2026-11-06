# Validation record

Hardware tests completed on 2026-11-06 on eight Hygon Z 100L GPUs. [summary.json](../validation/results/summary.json) records the results; the [HIP log](../validation/results/hip-extended.log) and [rocBLAS log](../validation/results/rocblas-numeric.log) contain the workload output.

## HIP and P 2P

The HIP workload allocates 72 device arrays (288 MiB payload), including 56 independent destination buffers, with one stream per destination GPU. Each of 3 rounds fills all devices, checks 56 directed copies of 4 MiB each, then checks 56 kernels reading peer source pointers. Every int 32 element is compared, and source arrays are checked again after the remote reads. All eight streams finish at each phase boundary before source buffers change.

Eight graphs capture fill+transform. Each of 3 replays first poisons the buffers, then checks every element. The final log reports CLEANUP_PASS with all tracked resource counters at zero.

## rocBLAS

The rocBLAS workload runs 64×64 column-major FP 32 SGEMM against a CPU reference. Small integer inputs make the expected results exactly representable in FP 32. Three SGEMM rounds and three graph replays run on all eight devices. Each iteration poisons the output and checks all 4096 values for finiteness and exact equality. The log contains 24 SGEMM and 24 graph pass records. Cleanup failures propagate through the exit code.

## Test setup

Recorded container attach, exit and removal codes were 0, with no OOM or State.Error. The runtime used the pinned community image's HIP 7.16.26385 and ROCr 1.21.0 libraries.

The topology adapter and C++ workloads match the hardware-tested source hashes in [source-provenance.json](source-provenance.json). The portable launcher changes storage paths and container names/labels; its changes are covered by mocked Docker lifecycle tests rather than a new GPU run. Seventeen unit tests cover topology checks and launcher behavior using synthetic fixtures.

Final host checks matched the driver, runtime and source hashes, plus retained weight metadata. Original-toolchain eight-card and 56-direction probes passed, and all cards returned to 2142208B of VRAM usage after KFD cleanup.

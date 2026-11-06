// Correctness-only HIP validation; compile this same source with the target HIP
// compiler/headers. No timing, cache/ISA-error suppression, or environment changes.
// Run under an EXTERNAL timeout: HIP synchronization/free/destroy can hang when
// the GPU/runtime is wedged. Neither cleanup nor the destructor explicitly syncs.
#include <hip/hip_runtime.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <vector>

namespace {
constexpr int kDevices = 8;
constexpr int kPairs = kDevices * (kDevices - 1);
constexpr int kRounds = 3;
constexpr int kReplays = 3;
constexpr std::size_t kDefaultN = std::size_t(1) << 20;
// Bound the optional n argument too: 72 device arrays <= 288 MiB and one
// reusable host array <= 4 MiB. Runtime/driver-internal allocations are separate.
constexpr std::size_t kMaxN = kDefaultN;
constexpr int kThreads = 256;
static_assert(sizeof(int) == 4, "This validator requires 32-bit int");
static_assert(std::numeric_limits<int>::max() == 2147483647,
              "This validator requires signed int32");

struct Failure {};

void check(hipError_t error, const char *call, int line) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "HIP_FAIL line=%d call=%s error=%d (%s)\n",
                     line, call, int(error), hipGetErrorString(error));
        throw Failure{};
    }
}
#define HIP_CHECK(call) check((call), #call, __LINE__)

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "VALIDATION_FAIL %s\n", message);
        throw Failure{};
    }
}

// All arithmetic stays within int32 even at kMaxN, seed=3, device=7:
// base <= 75,048,575; transform <= 225,145,851; negative fill >= -75,048,576.
__host__ __device__ int pattern(std::size_t i, int device, int seed) {
    return int(i) + device * 2000000 + seed * 20000000;
}
__host__ __device__ int transformed(int value, int dst) {
    return 3 * value + 7 + dst * 17;
}

__global__ void fill(int *out, std::size_t n, int device, int seed,
                     bool negative) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const int value = pattern(i, device, seed);
        out[i] = negative ? -value - 1 : value;
    }
}

__global__ void transform(int *out, const int *in, std::size_t n, int dst) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = transformed(in[i], dst);
}

struct DeviceResources {
    int *base = nullptr;
    int *out = nullptr;
    std::array<int *, kDevices> pair_out{}; // [src], allocated on this dst
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    bool capturing = false;
    bool work_pending = false;
};

struct Resources {
    std::array<DeviceResources, kDevices> devices{};
    std::array<std::array<bool, kDevices>, kDevices> owned_peer{};
    bool cleanup_attempted = false;

    bool has_pending_work() const noexcept {
        for (const auto &d : devices) if (d.work_pending) return true;
        return false;
    }

    static bool cleanup_check(hipError_t error, const char *what, int dst,
                              int src = -1) noexcept {
        if (error == hipSuccess) return true;
        std::fprintf(stderr, "CLEANUP_FAIL resource=%s dst=%d src=%d error=%d (%s)\n",
                     what, dst, src, int(error), hipGetErrorString(error));
        return false;
    }

    // Best effort, one attempt, retaining failed handles for residual reporting.
    // Do not retry cleanup in the destructor after an explicit cleanup failure.
    bool cleanup() noexcept {
        if (cleanup_attempted) return false;
        cleanup_attempted = true;
        const bool pending = has_pending_work();
        bool ok = !pending;
        if (pending) {
            // Failed drain does not prove remote readers stopped. Retain sources
            // and in-flight executables rather than risking use-after-free.
            std::fprintf(stderr, "CLEANUP_FAIL unresolved_work=1 retaining_resources_for_process_teardown=1\n");
        }
        for (int dst = kDevices - 1; !pending && dst >= 0; --dst) {
            auto &d = devices[dst];
            bool owns_anything = d.base || d.out || d.stream || d.graph ||
                                 d.executable || d.capturing;
            for (int src = 0; src < kDevices; ++src)
                owns_anything = owns_anything || d.pair_out[src] || owned_peer[dst][src];
            if (!owns_anything) continue;
            if (!cleanup_check(hipSetDevice(dst), "set_device", dst)) {
                ok = false;
                continue;
            }
            if (d.capturing) {
                // Abort/end an interrupted capture without executing the graph.
                const hipError_t e = hipStreamEndCapture(d.stream, &d.graph);
                if (cleanup_check(e, "end_capture", dst)) d.capturing = false;
                else ok = false;
            }
            if (d.executable) {
                if (cleanup_check(hipGraphExecDestroy(d.executable), "graph_exec", dst))
                    d.executable = nullptr;
                else ok = false;
            }
            if (d.graph) {
                if (cleanup_check(hipGraphDestroy(d.graph), "graph", dst)) d.graph = nullptr;
                else ok = false;
            }
            if (d.stream) {
                if (cleanup_check(hipStreamDestroy(d.stream), "stream", dst)) d.stream = nullptr;
                else ok = false;
            }
        }
        // All streams/graphs are disposed first, then allocations on ALL GPUs,
        // then owned peer edges. No source is intentionally freed mid-phase.
        for (int dst = kDevices - 1; !pending && dst >= 0; --dst) {
            auto &d = devices[dst];
            bool has_allocations = d.base || d.out;
            for (auto p : d.pair_out) has_allocations = has_allocations || p;
            if (!has_allocations) continue;
            if (!cleanup_check(hipSetDevice(dst), "set_device_free", dst)) {
                ok = false;
                continue;
            }
            for (int src = kDevices - 1; src >= 0; --src) {
                if (!d.pair_out[src]) continue;
                if (cleanup_check(hipFree(d.pair_out[src]), "pair_alloc", dst, src))
                    d.pair_out[src] = nullptr;
                else ok = false;
            }
            if (d.out) {
                if (cleanup_check(hipFree(d.out), "out_alloc", dst)) d.out = nullptr;
                else ok = false;
            }
            if (d.base) {
                if (cleanup_check(hipFree(d.base), "base_alloc", dst)) d.base = nullptr;
                else ok = false;
            }
        }
        for (int dst = kDevices - 1; !pending && dst >= 0; --dst) {
            bool has_edges = false;
            for (bool owned : owned_peer[dst]) has_edges = has_edges || owned;
            if (!has_edges) continue;
            if (!cleanup_check(hipSetDevice(dst), "set_device_peer", dst)) {
                ok = false;
                continue;
            }
            for (int src = kDevices - 1; src >= 0; --src) {
                if (!owned_peer[dst][src]) continue;
                if (cleanup_check(hipDeviceDisablePeerAccess(src), "owned_peer", dst, src))
                    owned_peer[dst][src] = false;
                else ok = false;
            }
        }
        int allocations = 0, streams = 0, graphs = 0, executions = 0, edges = 0, captures = 0;
        for (int dst = 0; dst < kDevices; ++dst) {
            const auto &d = devices[dst];
            allocations += int(d.base != nullptr) + int(d.out != nullptr);
            streams += int(d.stream != nullptr);
            graphs += int(d.graph != nullptr);
            executions += int(d.executable != nullptr);
            captures += int(d.capturing);
            for (int src = 0; src < kDevices; ++src) {
                allocations += int(d.pair_out[src] != nullptr);
                edges += int(owned_peer[dst][src]);
            }
        }
        ok = ok && allocations == 0 && streams == 0 && graphs == 0 &&
             executions == 0 && edges == 0 && captures == 0;
        std::printf("CLEANUP_%s remaining_allocs=%d streams=%d graphs=%d execs=%d owned_peers=%d captures=%d\n",
                    ok ? "PASS" : "FAIL", allocations, streams, graphs, executions, edges, captures);
        std::fflush(stdout);
        return ok;
    }

    ~Resources() { if (!cleanup_attempted) (void)cleanup(); }
    Resources() = default;
    Resources(const Resources &) = delete;
    Resources &operator=(const Resources &) = delete;
};

// Always visit every stream before reporting a failed boundary. Host reads and
// subsequent source writes happen ONLY after this entire boundary succeeds.
void sync_all(Resources &r, const char *phase, int round) {
    std::printf("BOUNDARY_BEGIN round=%d phase=%s streams=8\n", round, phase);
    std::fflush(stdout);
    bool ok = true;
    for (int dst = 0; dst < kDevices; ++dst) {
        hipError_t error = hipSetDevice(dst);
        if (error == hipSuccess) error = hipStreamSynchronize(r.devices[dst].stream);
        if (error != hipSuccess) {
            std::fprintf(stderr, "BOUNDARY_FAIL round=%d phase=%s dst=%d error=%d (%s)\n",
                         round, phase, dst, int(error), hipGetErrorString(error));
            ok = false;
        } else {
            r.devices[dst].work_pending = false;
        }
    }
    require(ok, "all-stream boundary failed");
    std::printf("BOUNDARY_PASS round=%d phase=%s streams=8\n", round, phase);
}

enum class Expected { Base, Negative, Transform };

void verify(std::vector<int> &host, const int *device_ptr, int dst, int src,
            int seed, Expected expected, const char *phase, int round) {
    HIP_CHECK(hipSetDevice(dst));
    HIP_CHECK(hipMemcpy(host.data(), device_ptr, host.size() * sizeof(int),
                        hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < host.size(); ++i) {
        int want = pattern(i, src, seed);
        if (expected == Expected::Negative) want = -want - 1;
        if (expected == Expected::Transform) want = transformed(want, dst);
        if (host[i] != want) {
            std::fprintf(stderr, "VALUE_FAIL round=%d phase=%s dst=%d src=%d index=%zu got=%d expected=%d\n",
                         round, phase, dst, src, i, host[i], want);
            throw Failure{};
        }
    }
}

void phase_pass(int round, const char *phase, int arrays, int pairs, std::size_t n) {
    std::printf("PHASE_PASS round=%d phase=%s devices=8 streams=8 arrays=%d pairs=%d checked_elements=%zu all_n=%zu\n",
                round, phase, arrays, pairs, std::size_t(arrays) * n, n);
    std::fflush(stdout);
}

void run(Resources &r, std::size_t n) {
    int count = 0, driver = 0, runtime = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    HIP_CHECK(hipDriverGetVersion(&driver));
    HIP_CHECK(hipRuntimeGetVersion(&runtime));
    std::printf("CONFIG devices=%d required=8 driver=%d runtime=%d n=%zu rounds=3 graph_replays=3 pair_count=56 correctness_only=1\n",
                count, driver, runtime, n);
    require(count == kDevices, "exactly eight visible devices required");
    const std::size_t bytes = n * sizeof(int);
    const unsigned int blocks = static_cast<unsigned int>((n + kThreads - 1) / kThreads);
    std::printf("MEMORY device_arrays=72 device_payload_bytes=%zu pair_payload_bytes=%zu host_payload_bytes=%zu\n",
                std::size_t(72) * bytes, std::size_t(56) * bytes, bytes);
    std::vector<int> host(n); // Only one reusable host buffer, never 56 copies.
    for (int dst = 0; dst < kDevices; ++dst) {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, dst));
        std::printf("DEVICE dst=%d name=%s arch=%s\n", dst, prop.name, prop.gcnArchName);
        HIP_CHECK(hipSetDevice(dst));
        auto &d = r.devices[dst];
        HIP_CHECK(hipStreamCreateWithFlags(&d.stream, hipStreamNonBlocking));
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d.base), bytes));
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d.out), bytes));
        for (int src = 0; src < kDevices; ++src) {
            if (src != dst)
                HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d.pair_out[src]), bytes));
        }
    }
    int available_pairs = 0, owned_pairs = 0, preexisting_pairs = 0;
    for (int dst = 0; dst < kDevices; ++dst) {
        HIP_CHECK(hipSetDevice(dst));
        for (int src = 0; src < kDevices; ++src) {
            if (src == dst) continue;
            int access = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&access, dst, src));
            require(access != 0, "every directed peer edge must be accessible");
            const hipError_t error = hipDeviceEnablePeerAccess(src, 0);
            if (error == hipSuccess) {
                r.owned_peer[dst][src] = true;
                ++owned_pairs;
            } else if (error == hipErrorPeerAccessAlreadyEnabled) {
                ++preexisting_pairs;
                // Acknowledge ONLY this expected API result, not other errors.
                const hipError_t last = hipGetLastError();
                if (last != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(last);
            } else HIP_CHECK(error);
            ++available_pairs;
        }
    }
    require(available_pairs == kPairs && owned_pairs + preexisting_pairs == kPairs,
            "56 directed peers required");
    std::printf("PEERS_PASS pairs=%d owned=%d preexisting=%d\n",
                available_pairs, owned_pairs, preexisting_pairs);
    std::fflush(stdout);

    for (int round = 1; round <= kRounds; ++round) {
        const int seed = round - 1;
        // Submit to all eight streams BEFORE any synchronization/host checking.
        for (int dst = 0; dst < kDevices; ++dst) {
            HIP_CHECK(hipSetDevice(dst));
            auto &d = r.devices[dst];
            d.work_pending = true;
            hipLaunchKernelGGL(fill, dim3(blocks), dim3(kThreads), 0, d.stream,
                               d.base, n, dst, seed, false);
            HIP_CHECK(hipGetLastError());
            hipLaunchKernelGGL(fill, dim3(blocks), dim3(kThreads), 0, d.stream,
                               d.out, n, dst, seed, true);
            HIP_CHECK(hipGetLastError());
        }
        sync_all(r, "fill", round);
        for (int dst = 0; dst < kDevices; ++dst) {
            verify(host, r.devices[dst].base, dst, dst, seed, Expected::Base, "fill_base", round);
            verify(host, r.devices[dst].out, dst, dst, seed, Expected::Negative, "fill_out", round);
        }
        phase_pass(round, "fill", 16, 0, n);

        // base pointers are read-only from here through remote_read hostcheck.
        // pair_out[dst][src] is disjoint from every base and every other pair.
        int copies = 0;
        for (int dst = 0; dst < kDevices; ++dst) {
            HIP_CHECK(hipSetDevice(dst));
            auto &d = r.devices[dst];
            d.work_pending = true;
            for (int src = 0; src < kDevices; ++src) {
                if (src == dst) continue;
                HIP_CHECK(hipMemcpyPeerAsync(d.pair_out[src], dst, r.devices[src].base,
                                             src, bytes, d.stream));
                ++copies;
            }
        }
        require(copies == kPairs, "copy submission count must be 56");
        sync_all(r, "peer_copy", round);
        int checked_pairs = 0;
        for (int dst = 0; dst < kDevices; ++dst) {
            for (int src = 0; src < kDevices; ++src) {
                if (src == dst) continue;
                verify(host, r.devices[dst].pair_out[src], dst, src, seed,
                       Expected::Base, "peer_copy", round);
                ++checked_pairs;
            }
        }
        require(checked_pairs == kPairs, "copy hostcheck count must be 56");
        phase_pass(round, "peer_copy", checked_pairs, checked_pairs, n);

        int remote_reads = 0;
        for (int dst = 0; dst < kDevices; ++dst) {
            HIP_CHECK(hipSetDevice(dst));
            auto &d = r.devices[dst];
            d.work_pending = true;
            for (int src = 0; src < kDevices; ++src) {
                if (src == dst) continue;
                hipLaunchKernelGGL(transform, dim3(blocks), dim3(kThreads), 0,
                                   d.stream, d.pair_out[src], r.devices[src].base, n, dst);
                HIP_CHECK(hipGetLastError());
                ++remote_reads;
            }
        }
        require(remote_reads == kPairs, "remote-read submission count must be 56");
        sync_all(r, "remote_read", round);
        checked_pairs = 0;
        for (int dst = 0; dst < kDevices; ++dst) {
            for (int src = 0; src < kDevices; ++src) {
                if (src == dst) continue;
                verify(host, r.devices[dst].pair_out[src], dst, src, seed,
                       Expected::Transform, "remote_read", round);
                ++checked_pairs;
            }
        }
        require(checked_pairs == kPairs, "remote-read hostcheck count must be 56");
        phase_pass(round, "remote_read", checked_pairs, checked_pairs, n);
        for (int dst = 0; dst < kDevices; ++dst)
            verify(host, r.devices[dst].base, dst, dst, seed,
                   Expected::Base, "source_unchanged", round);
        phase_pass(round, "source_unchanged", kDevices, 0, n);
        std::printf("ROUND_PASS round=%d copy_pairs=%d remote_pairs=%d all_n=%zu\n",
                    round, copies, remote_reads, n);
        std::fflush(stdout);
    }

    // All remote reads on ALL GPUs finished before graph capture can write base.
    // Capture two local kernels per device using ordinary HIP graph APIs.
    for (int dst = 0; dst < kDevices; ++dst) {
        HIP_CHECK(hipSetDevice(dst));
        auto &d = r.devices[dst];
        HIP_CHECK(hipStreamBeginCapture(d.stream, hipStreamCaptureModeGlobal));
        d.capturing = true;
        hipLaunchKernelGGL(fill, dim3(blocks), dim3(kThreads), 0, d.stream,
                           d.base, n, dst, kRounds, false);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(transform, dim3(blocks), dim3(kThreads), 0, d.stream,
                           d.out, d.base, n, dst);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamEndCapture(d.stream, &d.graph));
        d.capturing = false;
        require(d.graph != nullptr, "capture returned null graph");
        HIP_CHECK(hipGraphInstantiate(&d.executable, d.graph, nullptr, nullptr, 0));
        require(d.executable != nullptr, "instantiate returned null executable");
    }
    std::printf("GRAPH_CAPTURE_PASS devices=8 graphs=8 execs=8 kernels_per_graph=2\n");
    std::fflush(stdout);
    for (int replay = 1; replay <= kReplays; ++replay) {
        for (int dst = 0; dst < kDevices; ++dst) {
            HIP_CHECK(hipSetDevice(dst));
            auto &d = r.devices[dst];
            // Poison both arrays each replay so a no-op/stale graph cannot pass.
            // Stream order: poison -> captured fill -> captured local transform.
            d.work_pending = true;
            HIP_CHECK(hipMemsetAsync(d.base, 0xa5, bytes, d.stream));
            HIP_CHECK(hipMemsetAsync(d.out, 0xa5, bytes, d.stream));
            HIP_CHECK(hipGraphLaunch(d.executable, d.stream));
        }
        sync_all(r, "graph_replay", replay);
        for (int dst = 0; dst < kDevices; ++dst) {
            verify(host, r.devices[dst].base, dst, dst, kRounds,
                   Expected::Base, "graph_fill", replay);
            verify(host, r.devices[dst].out, dst, dst, kRounds,
                   Expected::Transform, "graph_transform", replay);
        }
        phase_pass(replay, "graph_replay", 16, 0, n);
    }
}
} // namespace

int main(int argc, char **argv) {
    // Line buffering keeps phase/boundary diagnostics visible under timeout.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::size_t n = kDefaultN;
    if (argc > 2) {
        std::fprintf(stderr, "USAGE: %s [n in 1..%zu]; rounds/replays fixed at 3\n", argv[0], kMaxN);
        return 2;
    }
    if (argc == 2) {
        const char *text = argv[1];
        bool digits = text[0] != '\0';
        for (const char *p = text; *p; ++p) digits = digits && *p >= '0' && *p <= '9';
        errno = 0;
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(text, &end, 10);
        if (!digits || errno == ERANGE || end == text || *end != '\0' ||
            parsed == 0 || parsed > kMaxN) {
            std::fprintf(stderr, "ARG_FAIL n must be a decimal integer in 1..%zu\n", kMaxN);
            return 2;
        }
        n = static_cast<std::size_t>(parsed);
    }
    Resources resources;
    bool passed = false;
    try {
        run(resources, n);
        passed = true;
    } catch (const Failure &) {
        // Detailed HIP/value error was already printed; cleanup still mandatory.
    } catch (const std::exception &e) {
        std::fprintf(stderr, "HOST_FAIL %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "HOST_FAIL unknown_exception\n");
    }
    if (resources.has_pending_work()) {
        // Explicit main-only recovery, NEVER destructor synchronization.
        // The external timeout must also cover this boundary.
        try {
            sync_all(resources, "failure_drain", 0);
        } catch (...) {
            std::fprintf(stderr, "FAILURE_DRAIN_FAIL cleanup_will_retain_unsafe_resources=1\n");
        }
    }
    const bool cleaned = resources.cleanup(); // Explicit terminal check gates PASS.
    if (!passed || !cleaned) {
        std::fprintf(stderr, "FAIL hip_extended validation=%s cleanup=%s\n",
                     passed ? "PASS" : "FAIL", cleaned ? "PASS" : "FAIL");
        return 1;
    }
    std::printf("PASS hip_extended devices=8 streams=8 rounds=3 copy_pairs_per_round=56 remote_pairs_per_round=56 graph_devices=8 graph_replays=3 all_n=%zu cleanup=PASS correctness_only=1\n", n);
    return 0;
}

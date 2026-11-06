# ML-Hygon-Z100L

Community ROCm 10.1 compatibility adapter for an eight-card Hygon Z100L system, inspired by [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906).

**已实现社区 ROCm 10.1 在八卡海光 Z100L 平台上的适配运行，完成计算、跨卡通信、HIP Graph 和 rocBLAS 数值验证。**

本项目提供海光 Z100L 的容器内拓扑适配、启动器和验证程序，使用上游社区的 ROCm 10.1 / gfx906 软件栈。拓扑覆盖随容器生效，退出后自动撤销。

## 为什么需要适配

原生 KFD 拓扑报告 gfx_target_version=90106。将其映射到经实测的 gfx906 ISA 后，新运行库可枚举设备，但首次 kernel 加载仍会失败：

~~~text
CREATE_QUEUE -> EFAULT
HSA executable freeze -> HSA_STATUS_ERROR_OUT_OF_RESOURCES
HIP -> 209 (no kernel image is available)
~~~

定位到海光驱动的严格 CWSR（计算上下文保存恢复区）尺寸检查：

| 属性 | 通用运行库计算值 | 海光驱动要求 |
|---|---:|---:|
| ctl_stack_size | 20480 / 0x5000 | 20480 / 0x5000 |
| cwsr_size | 36638720 / 0x22f1000 | 33550336 / 0x1fff000 |

[compat_topology.py](compat_topology.py) 根据已验证的硬件公式生成八份 properties 副本，[run_compatible.py](run_compatible.py) 将它们只读挂载到容器对应节点：

- gfx_target_version：90106 → 90006，显式使用经数值测试的 gfx906。
- cwsr_size：33550336。
- ctl_stack_size：20480。

社区运行库读取修正后的属性，按驱动要求完成内存分配、保存区初始化和队列创建。详见[适配原理](docs/compatibility.md)。

## 已验证平台

| 项目 | 环境 |
|---|---|
| GPU | 8 × Hygon Z100L，PCI 1d94:55b7 |
| 原生目标 / 验证 ISA | 90106 / gfx906:sramecc+:xnack-，wave64 |
| CU profile | 60 CU，64 KiB LDS，512 KiB VGPR，16 KiB SGPR，dp4x_status=0 |
| 驱动 | hydcu 6.2.26，srcversion B568DE62D6C1131D653C14F |
| 宿主 | Ubuntu 22.04，Linux 5.15.0-60-generic |
| 社区构建标注 | 10.1.0-gfx906+20261018172427 |
| 实际 HIP / ROCr | HIP 7.16.26385 / ROCr 1.21.0，Clang 24.0.0git |
| 容器用户态 | Ubuntu 24.04 / glibc 2.39 |

启动器检查八卡数量、PCI/CU profile、驱动版本及模块源身份，匹配后生成拓扑覆盖。

## 快速开始

环境要求：Python 3.10+、本机 Docker，以及 Docker、设备节点和 KFD 拓扑的访问权限。Python 部分仅依赖标准库。

1. 获取本仓库和固定镜像：

~~~bash
git clone https://github.com/ZH40s05/ML-Hygon-Z100L.git
cd ML-Hygon-Z100L
docker pull mixa3607/rocm-gfx906@sha256:2c124ecab56dd5f509a4310507435bc75588b3fed2e1d4fa28671a2977406395
mkdir -p work
cp validation/hip_extended.cpp validation/rocblas_numeric.cpp work/
~~~

镜像 ID 为 sha256:c5632b04e8874537ffb597c2037179fc62d2f7b7c8ad77adab342d3ef1109ef2。启动器会核对该 ID；镜像来源和摘要见[来源记录](docs/source-provenance.json)。

2. 在新 SDK 内编译和运行 HIP 扩展验证：

~~~bash
python3 -B run_compatible.py --case build-hip --timeout 240 -- \
  /opt/rocm/core-10.1/bin/hipcc -O2 -std=c++17 -Wall -Wextra \
  --offload-arch=gfx906 /work/hip_extended.cpp -o /work/hip-extended
python3 -B run_compatible.py --case verify-hip --timeout 240 -- /work/hip-extended
~~~

3. 编译和运行 rocBLAS 验证：

~~~bash
python3 -B run_compatible.py --case build-blas --timeout 240 -- \
  /opt/rocm/core-10.1/bin/hipcc -O2 --offload-arch=gfx906 \
  -I/opt/rocm/core-10.1/include -L/opt/rocm/core-10.1/lib \
  -Wl,-rpath,/opt/rocm/core-10.1/lib /work/rocblas_numeric.cpp \
  -lrocblas -o /work/rocblas-numeric
python3 -B run_compatible.py --case verify-blas --timeout 240 -- /work/rocblas-numeric
~~~

每次运行写入独立的 runs/case-UUID 目录，包含真实参数、拓扑覆盖、容器状态、输出和删除结果。work 和 runs 都在本地 checkout 下并被 Git 忽略。宿主 checkout 路径不要包含逗号，以免与 Docker --mount 参数格式冲突。

容器只读、无网络、cap-drop ALL、no-new-privileges；限制 CPU/内存/PID，memlock unlimited；仅 work 宿主目录以可写方式挂载，另提供临时 /tmp tmpfs；八项拓扑覆盖只读。

## 已完成验证

硬件验证日期：2026-11-06。

| 验证 | 已记录结果 |
|---|---|
| 默认 code-object / v4 单卡 | 全部通过 |
| 八卡逐卡整数计算 | 全部通过 |
| 56 个有向 P2P 对 | 复制与 GPU 远端 kernel 读通过 |
| 扩展并发测试 | 8 streams，3 轮，每对 4 MiB，56 向复制与远端读逐元素检查通过 |
| HIP Graph | 八卡各捕获两个 kernel，3 次回放；每次 poison 后逐元素检查通过 |
| rocBLAS SGEMM / Graph | 八卡各 3 轮 64×64 SGEMM 与 3 轮 Graph，每次 poison 输出并与 CPU 参考精确比较 |

完整测试规模和程序输出见[验证记录](docs/validation.md)。

在普通 CPU 机器上可运行白名单与容器生命周期测试：

~~~bash
PYTHONPATH=. python3 -B -m unittest discover -s tests -v
~~~

单元测试使用合成拓扑，覆盖设备检查与容器生命周期。

## 运行管理

启动器通过镜像 ID 和 run label 识别并清理本次容器。若创建超时，按 runs 中的 command/result 记录检查对应容器并完成清理。

后续工作：模型集成、性能测试与长时稳定性验证。

## 来源与许可

- 命名及社区 gfx906 软件栈参考：[mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)，固定参考[ROCm 目录](https://github.com/mixa3607/ML-gfx906/tree/20261031203345/rocm)。
- ROCr 属性支持参考固定版本的[上游源码](https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime)。
- 适配、启动和验证代码采用 [MIT License](LICENSE)；外部项目及镜像遵循各自许可。

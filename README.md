# ML-Hygon-Z100L

Community ROCm 10.1 compatibility adapter for an eight-card Hygon Z100L system, inspired by [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906).

**已实现社区 ROCm 10.1 在八卡海光 Z100L 平台上的适配运行，完成计算、跨卡通信、HIP Graph 和 rocBLAS 数值验证。** 当前为固定硬件与驱动组合的兼容原型；尚未进行本适配下的全模型性能或长期稳定性验证。

本项目提供容器内拓扑兼容层和正确性验证程序。保留宿主海光驱动及原运行时，容器退出后覆盖自动撤销。ROCm 10.1 社区构建与 gfx906 支持来自上游社区项目；本仓库增加海光 Z100L 的适配与验证，不包含整套 ROCm 构建或镜像。

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

原版社区运行库在分配内存、初始化保存区头之前读到正确属性。适配没有在 ioctl 提交时篡改尺寸，没有关闭内核检查，也没有修改驱动、固件或 HIP/HSA 库。详见[机制与边界](docs/compatibility.md)。

## 已验证平台

| 项目 | 固定范围 |
|---|---|
| GPU | 8 × Hygon Z100L，PCI 1d94:55b7 |
| 原生目标 / 验证 ISA | 90106 / gfx906:sramecc+:xnack-，wave64 |
| CU profile | 60 CU，64 KiB LDS，512 KiB VGPR，16 KiB SGPR，dp4x_status=0 |
| 驱动 | hydcu 6.2.26，srcversion B568DE62D6C1131D653C14F |
| 宿主 | Ubuntu 22.04，Linux 5.15.0-60-generic |
| 社区构建标注 | 10.1.0-gfx906+20261018172427 |
| 实际 HIP / ROCr | HIP 7.16.26385 / ROCr 1.21.0，Clang 24.0.0git |
| 容器用户态 | Ubuntu 24.04 / glibc 2.39 |

白名单要求恰好八卡、固定 PCI/CU profile、驱动版本与模块源身份，拒绝其他组合和冲突属性。它不是面向任意海光卡、任意驱动的通用兼容保证。

## 快速开始

在上述匹配平台上使用 Docker。Python 3.10+ 仅依赖标准库。Docker daemon 必须运行在同一台设备宿主机；用户需具备 Docker、设备节点和 KFD 拓扑读取权限。不要为运行本项目更换驱动或固件。

1. 获取本仓库和固定镜像：

~~~bash
git clone https://github.com/ZH40s05/ML-Hygon-Z100L.git
cd ML-Hygon-Z100L
docker pull mixa3607/rocm-gfx906@sha256:2c124ecab56dd5f509a4310507435bc75588b3fed2e1d4fa28671a2977406395
mkdir -p work
cp validation/hip_extended.cpp validation/rocblas_numeric.cpp work/
~~~

启动器验证镜像 ID 为 sha256:c5632b04e8874537ffb597c2037179fc62d2f7b7c8ad77adab342d3ef1109ef2，拒绝继承的运行时覆盖和未经验证的镜像配置。镜像来源及固定摘要见[来源记录](docs/source-provenance.json)。

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

容器只读、无网络、cap-drop ALL、no-new-privileges；限制 CPU/内存/PID，memlock unlimited；仅 work 目录可写，八项拓扑覆盖只读。正式验证不使用 LD_PRELOAD、HSA_OVERRIDE_GFX_VERSION 或宿主旧 HIP/HYHAL 注入。

## 已完成验证

硬件验证日期：2026-11-06。这里报告正确性，不报告性能：

| 验证 | 已记录结果 |
|---|---|
| 默认 code-object / v4 单卡 | 全部通过 |
| 八卡逐卡整数计算 | 全部通过 |
| 56 个有向 P2P 对 | 复制与 GPU 远端 kernel 读通过 |
| 扩展并发测试 | 8 streams，3 轮，每对 4 MiB，56 向复制与远端读逐元素检查通过 |
| HIP Graph | 八卡各捕获两个 kernel，3 次回放；每次 poison 后逐元素检查通过 |
| rocBLAS SGEMM / Graph | 八卡各 3 轮 64×64 SGEMM 与 3 轮 Graph，每次 poison 输出并与 CPU 参考精确比较 |

[验证记录](docs/validation.md)解释测试规模、原始程序输出与证据范围。适配器及两个 C++ 验证源与硬件实测版本逐字一致；公开启动器将原实验绝对路径改为 checkout-relative 路径，并更换容器名/标签，已做模拟 Docker 的单元测试，尚未对该便携导出启动器重新进行 GPU 实测。

在普通 CPU 机器上可运行白名单与容器生命周期测试：

~~~bash
PYTHONPATH=. python3 -B -m unittest discover -s tests -v
~~~

测试使用明确标记的合成拓扑，不依赖原实验机器或其目录。

## 清理与限制

启动器仅在镜像 ID 和本次 run label 匹配时删除容器。create 超时而 daemon 迟到完成时，单次 inspect 可能未见容器，结果会提示需要后续核对；不承诺所有异常自动零残留。异常后按 runs 内 command/result 记录核对确切容器名与 run label，仅清理身份匹配的容器。

本次实验最终核对：原驱动/HYHAL、两套旧运行时及原源码哈希一致；权重元数据与已有证书一致；旧工具链八卡和 56 向回归通过，显存恢复基线，没有本轮容器或探针残留。没有执行重启、驱动重载或 GPU reset。

尚未覆盖完整 ROCm 组件、所有 BLAS 形状与精度、RCCL、框架集成、全模型推理或长时稳定性。P2P API 与远端读正确性不代表已证明物理互联路由或带宽。未产生本适配下的 DSV41 性能分数。

## 来源与许可

- 命名及社区 gfx906 软件栈参考：[mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)，固定参考[ROCm 目录](https://github.com/mixa3607/ML-gfx906/tree/20261031203345/rocm)。
- ROCr 属性支持调查使用固定[候选上游源码](https://github.com/ROCm/rocm-systems/tree/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime)；其版本吻合，但未证明与镜像二进制逐字同源。
- 本仓库仅包含自行整理的适配、启动、测试及验证代码、说明和可公开的程序输出；不再分发海光厂商源码、驱动、固件、ROCm 二进制或模型权重。
- 本仓库代码使用 [MIT License](LICENSE)；外部项目和镜像遵循各自许可。无 AMD 或海光官方认证或背书。

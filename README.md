# MeloStudio

MeloStudio 是一个正在开发的歌声编辑与合成项目。本仓库目前包含跨平台 C++ 核心库的基础工程，以及可独立运行的 Python NHN 风格声码器。编辑器界面、独立核心进程、跨进程协议和插件宿主仍处于设计或后续实现阶段；本仓库目前不能直接启动完整的桌面编辑器。

## 仓库内容

| 目录 | 当前内容 | 详细说明 |
| --- | --- | --- |
| [`Melo.Core/`](Melo.Core/) | C++20 静态库：版本信息、强类型实体 ID 与时间类型，以及基础测试；支持 CMake 安装与引用 | [核心库 README](Melo.Core/README.md) |
| [`Melo.Vocoder/`](Melo.Vocoder/) | Python 声码器：LLSM 特征分析、数据预处理、训练、推理、模型导出与 SDK | [声码器 README](Melo.Vocoder/README.md) |

两个模块当前分别构建和运行。核心库尚未集成声码器，也没有可运行的 UI 或插件进程。未来的模块边界与接入方式见[核心层架构提案](Melo.Core/docs/core-architecture.md)；它是设计文档，不代表这些功能已经实现。

## 环境要求

| 模块 | 要求 |
| --- | --- |
| Melo.Core | CMake 3.28+、Ninja、支持 C++20 的编译器 |
| Melo.Vocoder | Python 3.12、[uv](https://docs.astral.sh/uv/)；训练与推理依赖 PyTorch，音频特征提取还需安装 `analysis` 可选依赖 |

两个模块没有统一的根目录安装命令。以下命令均从仓库根目录开始执行。

## 快速开始

### 构建与测试 Melo.Core

```bash
cd Melo.Core
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

`debug` 和 `release` 预设使用 Ninja。安装后，其他 CMake 项目可通过 `find_package(MeloCore CONFIG REQUIRED)` 与 `Melo::Core` 引用此库。详见[核心库使用说明](Melo.Core/README.md)。

### 安装与使用 Melo.Vocoder

```bash
cd Melo.Vocoder
uv sync --dev --extra analysis
uv run nhn-vocoder --help
uv run python -c "from melo.vocoder import VocoderSession"
```

`analysis` 提供从 WAV/FLAC 提取 72 维 LLSM 特征所需的依赖。如果只使用已有特征进行训练或推理，可以先运行 `uv sync --dev`。NVIDIA CUDA 训练需要单独安装匹配驱动的 PyTorch wheel；操作步骤见[声码器 README 的 PyTorch 章节](Melo.Vocoder/README.md#pytorch-cpu-与-cuda-版本)。

Python 导入包为 `melo.vocoder`，源码位于 `Melo.Vocoder/src/`。安装会创建项目内的 `.venv`，不依赖仓库根目录的旧环境。

以下示例假定你已准备人声音频，并在 `Melo.Vocoder/` 目录执行：

```bash
# 将原始音频转换为配对的 LLSM 特征与 48 kHz 目标音频
uv run nhn-vocoder preprocess raw_wavs data/prototype_5_10h --f0-backend fcpe --f0-device cpu

# 使用示例配置训练；配置中的数据路径对应上一步的输出
uv run nhn-vocoder train --config configs/nhn-prototype-5-10h.yaml

# 用训练得到的 checkpoint 和 72 维特征生成音频
uv run nhn-vocoder infer input.npy checkpoints/nhn-prototype-5-10h/best.pt output.wav --device cpu
```

配置式训练的路径取决于 YAML 内容；推理命令中的 checkpoint 路径仅为示例，应替换为实际训练产物。仓库没有附带可直接用于合成的预训练权重。更多预处理、带宽扩展、SDK、导出与基准命令见[声码器完整说明](Melo.Vocoder/README.md)。

## 当前实现状态

- **核心库**：已有可编译、可测试、可安装的 C++20 基础结构；项目模型、编辑事务、撤销重做、播放与跨进程服务尚未实现。
- **声码器**：已有 72 维 LLSM 输入、48 kHz 波形输出的模型代码，以及预处理、训练、推理和基础导出流程；音质需要使用配对数据训练并验证。
- **集成应用**：UI、核心进程协议、插件宿主及声码器与核心的正式集成仍在规划中。

声码器的默认特征顺序为 `VUV(1) + F0(1) + Rd(1) + 谱编码(64) + BAP(5)`，共 72 维；默认采样率为 48 kHz、帧移为 256 个采样点。具体输入形状、配置与 SDK 契约见[声码器 README](Melo.Vocoder/README.md#输入与输出)。

## 测试

```bash
cd Melo.Core
ctest --preset debug
```

```bash
cd Melo.Vocoder
uv sync --dev
uv run pytest
```

运行 C++ 测试前须先完成上面的 CMake 配置与构建。Python 测试可能根据所装可选依赖和设备能力跳过部分检查。

## 文档索引

- [核心层架构文档导航](Melo.Core/docs/README.md)：架构提案、目录规范、决策记录与渲染协议设计。
- [声码器完善路线](Melo.Vocoder/docs/nhn-vocoder-roadmap.md)：已实现能力、实验项与验证计划。
- [5～10 小时原型训练指南](Melo.Vocoder/docs/prototype-training-5-10h.md)：小规模数据验证流程。
- [训练数据与硬件建议](Melo.Vocoder/docs/training-data-and-hardware.md)：采集规模、磁盘与训练设备规划。

## 许可证

仓库目前没有项目级 `LICENSE` 文件。声码器的可选分析依赖 `pyllsm2/libllsm2` 涉及 GPL-3.0-or-later；分发或商用前请核查依赖许可证，详见[声码器许可证说明](Melo.Vocoder/README.md#性能说明)。

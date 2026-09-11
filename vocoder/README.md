# NHN vocoder

这是一个面向 72 维 LLSM 条件特征的轻量、非自回归 harmonic-plus-noise
声码器实现。它依据参考图实现了非因果门控卷积、三路动态 FIR 频谱头、十段噪声控制和
16 子带 PQMF 后处理，并加入 pyllsm2 分析、数据集、训练、checkpoint 与推理入口。

## 输入与输出

每帧恰好 72 维：

| 索引 | 内容 | 约定范围 |
| --- | --- | --- |
| 0 | VUV 清/浊音 | 0 或 1 |
| 1 | F0（Hz） | 0–2000（无声段可为 0） |
| 2 | Rd 声门参数 | 0.02–3.0 |
| 3–66 | pyllsm2 Coder 的 64 维谱编码 | 默认归一化范围 -100–40 |
| 67–71 | 5 段 BAP | 0–1 |

输入形状可以是 `[T,72]`、`[B,T,72]` 或 `[B,72,T]`。默认 48 kHz、hop=256，
输出形状为 `[B,1,T*256]`。若上游特征使用不同的帧移，只需同步修改
`NHNVocoderConfig.hop_length`；训练 WAV 的采样率必须与配置一致。

## 使用

项目使用 Python 3.12 和 uv。先进入 `vocoder/` 项目目录；目录内的 `.venv`
符号链接会复用仓库根目录现有的 Python 3.12 环境：

```bash
cd vocoder
uv sync --dev
```

训练配置按职责拆为两个文件：[基模配置](configs/nhn-base.yaml) 声明
`config_type: nhn_base_train`，[DSP 后训练配置](configs/nhn-dsp-post.yaml) 声明
`config_type: nhn_dsp_post_train`。加载器会核对类型，防止两个训练阶段误用配置；文件内
的相对路径以 YAML 所在目录为基准。训练命令为：

```bash
uv run nhn-train --config configs/nhn-base.yaml
uv run nhn-train --config configs/nhn-dsp-post.yaml
```

命令行参数优先于 YAML，因此临时降低显存占用无需改文件：

```bash
uv run nhn-train --config configs/nhn-base.yaml --batch-size 1 --segment-seconds 1
```

分析 WAV 还需要 pyllsm2 和外部 F0 分析器：

```bash
CFLAGS="-Wno-error=implicit-function-declaration" uv sync --extra analysis
```

在 macOS 上，如果当前 pyllsm2 版本被 Apple Clang 的隐式声明检查拦截，可使用：

```bash
CFLAGS="-Wno-error=implicit-function-declaration" \
  python -m pip install pyllsm2==0.2.0
```

pyllsm2 自身不包含 F0 提取。本项目默认使用更适合歌声的 FCPE，也可选择 RMVPE
或与 pyllsm2 官方测试一致的 Praat/Parselmouth。F0 会对齐到 libllsm2 的帧移，
然后交给原生 `Coder(order_spec=64, order_bap=5)`；编码结果正好是
`3 + 64 + 5 = 72` 维。批量分析目录：

推荐直接使用统一预处理命令。它会递归扫描原始 WAV，将配对 WAV/NPY 写入训练目录，
最后生成 `feature_stats.npz` 和 `preprocess_report.json`：

```bash
uv run nhn-preprocess raw_wavs data/train --f0-backend fcpe --f0-device cpu
```

支持中断后重跑：已经存在的 NPY 会跳过，但仍会重新汇总全部成功文件的统计信息。
需要强制重算时添加 `--overwrite`，任意文件失败时立即停止可添加 `--fail-fast`。

下面的 `nhn-analyze` 和 `nhn-stats` 保留给需要单独调试某一步的场景：

```bash
python -m vocoder.analyze data/train --sample-rate 48000 --hop-length 256
```

选择 F0 后端：

```bash
python -m vocoder.analyze data/train --f0-backend fcpe --f0-device cuda
python -m vocoder.analyze data/train --f0-backend rmvpe
python -m vocoder.analyze data/train --f0-backend parselmouth
```

- `fcpe`：默认，针对单声道歌声，速度快，对快速音高变化较友好。
- `rmvpe`：噪声或残留伴奏条件下通常更稳，本实现采用 ONNX CPU 推理。
- `parselmouth`：依赖最轻、行为容易复现，适合作为兼容和排错后端。

该命令递归查找 WAV，并在每个 WAV 旁生成同名 NPY。已有 NPY 默认跳过；需要重做时
添加 `--overwrite`。单文件和独立输出目录也受支持：

```bash
python -m vocoder.analyze input.wav --output input.npy
python -m vocoder.analyze raw_wavs --output data/train
```

普通预处理支持 PCM/float WAV、FLAC、自动混为 mono，并会把非 48 kHz 输入自动重采样。
训练数据为同名文件对，例如 `001.npy`（`float32 [T,72]`）与 `001.wav`
（48 kHz mono target）：

```bash
uv run nhn-train data/train checkpoints/run1 \
  --batch-size 8 --segment-seconds 2 --device cpu
```

需要让同一个模型在推理时接受低于 48 kHz 的来源并补全到 48 kHz，可从 48 kHz 母带
建立多采样率 BWE 数据集：

```bash
uv run nhn-preprocess-bwe masters_48k data/bwe_train \
  --source-sample-rates 8000,12000,16000,22050,24000,32000,44100,48000 \
  --f0-backend fcpe --f0-device cpu

uv run nhn-train data/bwe_train checkpoints/bwe1 --device cuda
```

该数据集的训练 target 始终为原始 48 kHz WAV；不同来源采样率只改变输入 LLSM。
预处理会生成分组 train/valid manifest、统计和质量报告，训练器会自动识别。

对照 FCPE 与 RMVPE：

```bash
uv run nhn-compare-f0 raw_audio reports/f0-comparison.json \
  --first fcpe --second rmvpe
```

训练先使用 waveform/STFT/Mel/相位/瞬时频率/F0 谐波复合损失；默认从第 10,000
step 开启轻量 HiFi-GAN 风格 MPD/MSD、LSGAN 和 feature matching：

```bash
uv run nhn-train data/train checkpoints/run1 \
  --gan-start-step 10000 --adversarial-weight 1 --feature-matching-weight 2
```

MPD/MSD 只参与训练，不会增加 `NHNVocoder` 的推理参数量。对很小的数据做快速
过拟合时，可将 `--gan-start-step` 调到 100–1000；从第 0 step 启用通常不稳定。

如果数据目录含统一预处理生成的 `feature_stats.npz`，训练会自动加载，无需再传
`--feature-stats`；也可以显式指定另一份统计文件覆盖自动选择。

训练目录会生成 `latest.pt`、`best.pt`、`training.jsonl` 和 `tensorboard/`。
断点续训：

```bash
uv run nhn-train data/train checkpoints/run1 \
  --resume checkpoints/run1/latest.pt --epochs 200
```

CUDA 训练可添加 `--device cuda --amp`；禁用混合精度使用 `--no-amp`。没有独立
验证目录时，默认固定抽取 5% 文件作为验证集，也可用
`--validation-data data/valid`。

推理。旧入口继续支持 NPY；统一入口还支持 NPZ、字段 mapping 的 SDK 调用，以及
WAV/FLAC 音频分析。音频输入允许不高于 48 kHz，输出始终取 checkpoint 的 48 kHz：

```bash
uv run nhn-infer example.npy checkpoints/run1/best.pt outputs/output.wav --device cpu

uv run nhn-synthesize input_16k.wav checkpoints/bwe1/best.pt outputs/output_48k.wav \
  --device cpu --f0-backend fcpe --chunk-frames 750 --overlap-frames 32
```

噪声分支由 `--seed` 固定，因此同一特征和 checkpoint 可以得到可复现结果。
Python SDK：

```python
from vocoder import LLSMFeatures, VocoderSession

session = VocoderSession.from_checkpoint("checkpoints/bwe1/best.pt", device="cpu")
features = LLSMFeatures.from_numpy(values)  # [T, 72] 或 [72, T]
audio = session.synthesize_chunked(features, chunk_frames=750, overlap_frames=32)
audio.save("output_48k.wav")

# 音频入口会重采样、提取 F0/LLSM，并检查 checkpoint 的 BWE metadata
audio = session.synthesize_waveform("input_16k.wav", f0_backend="fcpe")
```

部署导出与目标机器基准：

```bash
uv sync --extra export
uv run nhn-export checkpoints/bwe1/best.pt deploy/inference.pt --format checkpoint
uv run nhn-export deploy/inference.pt deploy/model.ts --format torchscript --frames 375
uv run nhn-export deploy/inference.pt deploy/model.onnx --format onnx --frames 375
uv run nhn-benchmark deploy/inference.pt --seconds 2 --runs 10 --device cpu
```

TorchScript 可接收不同帧数。当前 ONNX 导出固定 `batch=1` 和 `--frames` 指定的长度；
长音频使用 SDK 分块。float16/bfloat16/int8 是否可用由基准程序按目标设备和算子组合
实际检测，不支持时会明确报告。完整接口边界见
[输入扩展与 SDK 规划](docs/vocoder-input-sdk-plan.md)。

## P5 DSP 控制与后训练

SDK 现在支持在稳定 LLSM72 基模之上添加帧级 DSP 控制，并可冻结基模、只训练小型
自动控制头：

```bash
uv run nhn-train --config configs/nhn-dsp-post.yaml
```

使用统一配置时只需：

```bash
uv run nhn-train --config configs/nhn-dsp-post.yaml
```

普通 checkpoint 默认完全旁路 DSP；后训练 checkpoint 会自动加载控制头；上层也可用
`SynthesisRequest` 显式提供控制。设计、范围和 SDK 示例见
[P5 DSP 与上层后训练](docs/p5-dsp-post-training.md)。

## 性能说明

- 默认网络按参考图的通道数设计，float32 权重约 12 MB；实际 checkpoint 还会包含
  固定 PQMF/噪声滤波 buffer；带两个 optimizer 和判别器的训练 checkpoint 会显著更大。
- 当前 16 子带 PQMF 使用 254-tap 抗混叠滤波器。本机本轮基准生成 2 秒/48 kHz
  条件约 0.136 秒（RTF 约 0.068）；实际速度需在目标机器用训练后模型复测。
- 网络是一次性并行生成而非逐采样自回归，CPU 通常会很有竞争力。是否能达到
  “2 秒音频 CPU 比 GPU 快”取决于 CPU、PyTorch 构建、线程数和传输开销，不能由
  架构本身保证。
- 8 GB 是运行时激活/训练图的预算描述，不是模型大小；SDK 使用
  `torch.inference_mode()`，长音频可用 `synthesize_chunked` 重叠分块。
- 未训练 checkpoint 只会产生近静音/噪声，必须用配对 LLSM/WAV 数据训练后才有
  可用音质。
- pyllsm2/libllsm2 使用 GPL-3.0-or-later；如果项目计划闭源或商业分发，请先评估
  其许可证影响，或向上游作者咨询替代许可。

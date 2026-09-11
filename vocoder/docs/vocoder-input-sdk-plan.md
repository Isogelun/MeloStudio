# NHN Vocoder 输入扩展与 SDK 规划

## 目标与边界

实现状态：P3 的多格式读取、自动重采样、多采样率训练对、manifest、质量报告和 F0
后端比较，以及 P4 的 `VocoderSession`、adapter registry、低采样率 WAV 统一推理、
重叠分块、纯 checkpoint 和 TorchScript/ONNX 导出均已落地。流式迭代器与取消机制
仍是后续服务化扩展，不属于本轮基础部署接口。

最终采样率约定：

```text
训练母带/target：固定 48 kHz
模型内部分析时间轴：固定 48 kHz
推理音频输入：允许 <= 48 kHz
模型/WAV 输出：固定 48 kHz
```

低采样率输入先统一到 48 kHz 时间轴，再由同一个经过混合带宽训练的 checkpoint 补全
缺失频带；48 kHz 输入不需要换模型。16 kHz 是常见示例，不是单独模式。

核心模型继续只接受规范化之前的 72 维 LLSM 帧，避免把文件读取、F0 提取、重采样、
上层模型字段映射和长音频分块混进神经网络 `forward`。扩展能力放在独立 SDK 层：

```text
文件/数组/上层模型输出/WAV
          ↓ FeatureAdapter
CanonicalLLSMFeatures [T,72] + metadata
          ↓ VocoderSession
NHNVocoder
          ↓
AudioResult [samples] + sample_rate
```

这样上层应用只依赖稳定 SDK；模型内部结构、checkpoint 加载和设备选择可以继续演进。

## 唯一标准输入

定义 `CanonicalLLSMFeatures`，内部数据固定为 contiguous `float32 [T,72]`：

| 索引 | 字段 | 数量 |
| --- | --- | ---: |
| 0 | VUV | 1 |
| 1 | F0 Hz | 1 |
| 2 | Rd | 1 |
| 3–66 | spectral envelope coding | 64 |
| 67–71 | BAP | 5 |

同时携带但不送入网络的 metadata：`sample_rate`、`hop_length`、`frame_count`、
`source_id`、`feature_version`。输入帧率不是独立配置，而是
`sample_rate / hop_length`；默认配置为 48,000/256 = 187.5 帧/秒，即每帧约
5.333 ms。

校验器负责：维度/布局转换、float32/contiguous、NaN/Inf、VUV/F0/Rd/BAP 范围、
空输入、metadata 与 checkpoint 配置一致性。默认严格报错；上层可以显式选择
`validation="repair"` 做裁剪或补齐，并接收 warning 列表。

## 第一版 Python SDK 接口

建议新增 `vocoder/sdk/`，公开以下接口：

```python
from vocoder.sdk import VocoderSession, LLSMFeatures, AudioResult

session = VocoderSession.from_checkpoint(
    "best.pt",
    device="cpu",
    seed=1234,
)

features = LLSMFeatures.from_numpy(values)       # [T,72] 或 [72,T]
result = session.synthesize(features)            # AudioResult
result.save("output.wav")

print(result.sample_rate)                        # 48000（取自 checkpoint）
print(result.samples.shape)                      # [T * hop_length]
```

核心类型草案：

```python
@dataclass(frozen=True)
class LLSMFeatures:
    values: np.ndarray
    sample_rate: int | None = None
    hop_length: int | None = None
    source_id: str | None = None
    feature_version: str = "llsm72-v1"

@dataclass(frozen=True)
class AudioResult:
    samples: np.ndarray
    sample_rate: int
    channels: int = 1

class VocoderSession:
    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", seed=1234): ...
    def synthesize(self, features, *, seed=None) -> AudioResult: ...
    def synthesize_to_file(self, features, output, *, seed=None) -> AudioResult: ...
    def synthesize_batch(self, items, *, seeds=None) -> list[AudioResult]: ...
```

`VocoderSession` 必须只加载一次 checkpoint，并复用模型。随机噪声以 session seed
为默认值，单次调用可覆盖，从而兼顾复现和批量生成。

## 输入适配器

用协议而不是大量 `if isinstance` 扩展输入：

```python
class FeatureAdapter(Protocol):
    def can_handle(self, source: object) -> bool: ...
    def convert(self, source: object, context: AdapterContext) -> LLSMFeatures: ...
```

内置适配器按以下顺序实现：

1. `NumpyAdapter`：`np.ndarray`、`.npy`、`.npz`。
2. `TorchAdapter`：CPU/CUDA tensor，支持 `[T,72]`、`[B,T,72]`、`[B,72,T]`。
3. `MappingAdapter`：上层按字段给出 `vuv/f0/rd/spectral/bap`，SDK 负责拼成 72 维。
4. `WaveformAnalysisAdapter`：WAV/波形先经 FCPE 或 RMVPE，再经 pyllsm2 编码。
5. 后续 `AcousticModelAdapter`：将特定上层声学模型输出映射为 LLSM，不污染核心 SDK。

字段式调用示例规划：

```python
result = session.synthesize({
    "vuv": vuv,                    # [T] 或 [B,T]
    "f0": f0_hz,
    "rd": rd,
    "spectral_envelope": sp64,     # [...,T,64]
    "bap": bap5,                   # [...,T,5]
})
```

适配器注册接口预留为 `session.adapters.register(adapter, priority=...)`，便于 MeloStudio
上层以后添加自己的 acoustic-model 输出类型，而不修改声码器包。

## 采样率与长度规则

当前默认 checkpoint 的输出是 **48,000 Hz、单声道**。推理代码必须读取
`model.config.sample_rate` 写入 WAV，不能由调用者随意声明另一个采样率。

```text
输出采样数 = 输入帧数 T × checkpoint.hop_length
输出时长   = 输出采样数 / checkpoint.sample_rate
```

默认 `hop_length=256`。例如 375 帧输出 96,000 个采样，即 2 秒、48 kHz。

若未来需要 44.1 kHz 或 24 kHz，应训练或微调对应 `sample_rate/hop_length` 的 checkpoint；
SDK 可在生成之后提供显式 resample 工具，但不能把“重采样后的 WAV”误称为模型原生采样率。

## 训练 WAV 与模型输出

### 当前普通声码器模式

当前训练数据是一一对应的：

```text
item.npy  float32 [T,72]，从 item.wav 分析得到
item.wav  48,000 Hz、mono、当前读取器要求 16-bit PCM
```

训练目标是 `item.wav` 的真实波形，模型输出为：

```text
[B, 1, T * 256] float waveform
        ↓ 写文件
48,000 Hz、mono、16-bit PCM WAV
```

也就是说，当前配置下训练 WAV 是 48 kHz，推理输出同样是 48 kHz。训练器会检查 WAV
采样率；直接放入 16 kHz WAV 会报采样率不匹配，不会把它静默当成 48 kHz。

## 任意低采样率输入、固定 48 kHz 输出模式

修正后的目标是：**训练母带、模型时间轴和输出始终为 48 kHz；推理入口可以接受
低于或等于 48 kHz 的音频**。16 kHz 只是常见输入之一，也可以输入 8、12、22.05、
24、32、44.1 或 48 kHz。低于 48 kHz 时同时执行带宽扩展（BWE）；48 kHz 输入则走
普通分析重建路径。

### 推理数据流

```text
8–48 kHz mono WAV
    ↓ 检测原始采样率
    ↓ 高质量重采样到 48 kHz（只统一时间轴，不创造高频）
FCPE/RMVPE + pyllsm2
    ↓
低带宽来源的 LLSM72 [T,72]
    ↓ 统一 NHNVocoder vocoder+BWE checkpoint
48 kHz mono waveform
    ↓
48 kHz mono WAV
```

先重采样到 48 kHz 再提取 LLSM，是为了继续使用 `hop_length=256` 和每秒 187.5 帧，
让任意输入采样率都落到同一个模型时间轴。重采样不会生成源 Nyquist 以上的真实内容；
模型产生的高频来自训练数据学到的条件先验，因此是“合理补全”，不是恢复原录音中
已经丢失的精确高频。

### 正确训练数据

所有训练目标仍来自高质量 48 kHz 母带。每次训练可以随机选择完整路径或一种低采样率
退化路径，构造严格对齐的数据对：

```text
target_48k.wav
    ├─→ 作为训练目标：原始 48 kHz 波形
    ├─→ 原始 48 kHz → 提取完整 LLSM72
    └─→ 随机低通 → 降采样到 8/12/16/22.05/24/32/44.1 kHz
                 → 再重采样到 48 kHz
                 → 提取低带宽 LLSM72
```

落盘后仍可保持现有配对格式：

```text
item.npy  # 从本次选择的完整或低带宽分支提取的 [T,72]
item.wav  # 原始高质量 48 kHz target
```

这样训练器始终读取 48 kHz target，变化集中在预处理/在线退化阶段。不能用彼此无关的
低采样率音频和 48 kHz 音频配对；二者必须来自同一录音、起点一致、长度一致，且
下采样前必须低通以避免混叠。

### 使用一个覆盖全带宽和低带宽的 checkpoint

SDK metadata 增加：

```text
task_mode: "vocoder_bwe"
analysis_sample_rate: 48000
trained_input_sample_rates: [8000, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
output_sample_rate: 48000
```

同一个 checkpoint 混合学习完整 48 kHz 条件和多种低带宽条件，因此上层无需按输入
采样率手动换模型。metadata 记录训练覆盖的采样率；SDK 可以接受其他 `<48 kHz`
采样率，但应提示它不在已验证分布内。只用完整 48 kHz 条件训练的旧 checkpoint 仍能
接收上采样后的低采样率音频，但高频补全质量没有保证。

### SDK 调用规划

```python
session = VocoderSession.from_checkpoint("nhn-vocoder-bwe-48k.pt")

result = session.synthesize_waveform(
    input_samples,
    input_sample_rate=16000,
)

assert result.sample_rate == 48000
result.save("output_48k.wav")
```

若输入本来就是上层生成的 LLSM72，则直接调用 `session.synthesize(features)`，无需经过
WAV；上层应在 metadata 中声明原始带宽或原始采样率，让 SDK 能进行分布校验和诊断。

### 预处理入口规划

统一预处理命令增加显式模式，而不是自动猜测：

```bash
uv run nhn-preprocess-bwe raw_48k data/bwe_train \
  --source-sample-rates 8000,12000,16000,22050,24000,32000,44100,48000 \
  --analysis-sample-rate 48000 \
  --target-sample-rate 48000 \
  --f0-backend fcpe
```

报告中保存原始长度、各低采样率长度、回升采样长度、LLSM 帧数、目标长度、对齐误差、
低通参数和重采样器版本。任何一个文件出现超过一帧的对齐误差都应停止，而不是裁剪后
静默训练。

## BWE 数据目录与清单格式

建议不要只依赖同名文件隐式关联，预处理同时生成 JSONL manifest：

```text
data/bwe_train/
├── features/
│   └── singer_a/
│       ├── song_001@48000.npy
│       ├── song_001@16000.npy
│       └── song_001@8000.npy
├── targets/
│   └── singer_a/song_001.wav
├── manifests/
│   ├── train.jsonl
│   └── valid.jsonl
├── feature_stats.npz
└── preprocess_report.json
```

每条 manifest 记录：

```json
{
  "id": "singer_a/song_001",
  "features": "features/singer_a/song_001@16000.npy",
  "target": "targets/singer_a/song_001.wav",
  "frames": 1875,
  "target_samples": 480000,
  "source_sample_rate": 16000,
  "analysis_sample_rate": 48000,
  "target_sample_rate": 48000,
  "source_bandwidth_hz": 8000,
  "task_mode": "vocoder_bwe"
}
```

训练/验证划分必须按歌手或歌曲分组，不能将同一首歌的相邻切片同时放入训练集和验证集，
否则指标会虚高。同一 target 的不同采样率特征作为多条 manifest 记录，共用一份 48 kHz
target；数据加载器按 target id 分组采样，避免采样率变体较多的歌曲被重复过度训练。

## 重采样与对齐规范

所有低采样率路径都必须使用带抗混叠低通的高质量 resampler。以 16 kHz 为例：

```text
48 kHz target
  ↓ low-pass（截止频率不高于约 7.6–7.9 kHz，保留过渡带）
16 kHz source
  ↓ 带限插值 ×3
48 kHz analysis waveform
```

实现层预留 `ResamplerBackend`，首版选择一个确定性后端并把版本写入报告。不同后端的
群延迟和边缘填充可能不同，不能在同一个数据集中混用。预处理应进行脉冲对齐测试，记录
首个有效样点偏移；若后端引入固定延迟，应在提取特征前统一补偿。

长度采用目标驱动规则：

```text
target_frames = floor(target_samples / 256)
usable_target_samples = target_frames * 256
features.shape[0] 必须等于 target_frames
```

只允许裁掉目标尾部不足一帧的样点。禁止通过分别拉伸输入和目标来“凑长度”，因为这会
破坏 F0 相位和瞬态对齐。

## BWE 专用训练目标

普通全频谱损失仍然保留，但 BWE 需要单独观察低频保真与高频生成：

```text
L_total = L_base
        + λ_low  * L_lowband_consistency
        + λ_high * L_highband_spectral
        + λ_hadv * L_highband_adversarial
```

- `L_base`：现有 waveform、MR-STFT、Mel、phase、IF、harmonic、GAN 和 feature matching。
- `L_lowband_consistency`：将输出降回该样本的原始采样率，与低带宽 source 比较，防止
  模型在补高频时破坏已有语音主体。
- `L_highband_spectral`：从该样本的源 Nyquist 到 24 kHz，对输出与 target 的
  STFT/Mel 区域加权比较。
- `L_highband_adversarial`：可选的小型高频判别器，仅观察高通后的波形，降低高频出现
  固定嘶声或重复纹理的风险。

初始权重不要凭文档固定为最终值。建议先用 `λ_low=1.0`、`λ_high=1.0`、
`λ_hadv=0` 做重建预热，在全频 MPD/MSD 稳定后再开启高频判别器。所有分项损失必须
独立写入 TensorBoard/JSONL，便于确认“高频指标改善”没有以低频失真为代价。

## 训练课程与数据增强

建议分三阶段：

1. **普通 vocoder 预训练**：完整 48 kHz LLSM → 48 kHz target，学习稳定的声源与
   声道合成。
2. **混合带宽重建微调**：同时保留完整 48 kHz 条件，并加入随机多采样率退化条件 →
   原始 48 kHz target，先不开启 GAN。
3. **BWE 对抗微调**：开启 MPD/MSD；确认无固定高频噪声后，再评估高频判别器。

第二、三阶段不能丢掉 48 kHz 完整路径，否则模型可能为了低带宽补全而降低正常 48 kHz
输入的重建能力。初始建议一个 batch 中约 25–35% 使用 48 kHz 完整条件，其余样本在已
配置的低采样率中均衡采样；最终比例需要用各采样率验证集调整。

训练输入不能只使用一种理想低通。为了适应真实低采样率来源，可在保持 target 不变时
随机化 source 退化：

- 低通截止频率与过渡带；
- 轻微响度变化；
- 16-bit/24-bit 量化；
- 很轻的背景噪声；
- 可选的常见语音编码退化。

首版不加入混响、强降噪或音高变换，因为它们会把任务从带宽扩展扩大成语音增强/变声，
难以判断问题来源。验证集必须使用固定、确定性的退化配置。

## Checkpoint 配置草案

在 `NHNVocoderConfig` 或独立 inference metadata 中增加：

```json
{
  "feature_schema": "llsm72-v1",
  "task_mode": "vocoder_bwe",
  "trained_input_sample_rates": [8000, 12000, 16000, 22050, 24000, 32000, 44100, 48000],
  "analysis_sample_rate": 48000,
  "output_sample_rate": 48000,
  "sample_rate": 48000,
  "hop_length": 256,
  "channels": 1
}
```

加载 checkpoint 时，SDK 先检查 metadata，再选择对应适配器。若输入采样率未被训练
覆盖，SDK 默认给出分布外 warning；严格模式可以拒绝。48 kHz 输入和受支持的低采样率
输入都使用同一个 `vocoder_bwe` checkpoint。

## SDK 分层与实际文件结构

```text
vocoder/
├── sdk/
│   ├── __init__.py
│   ├── session.py          # VocoderSession
│   ├── types.py            # LLSMFeatures / AudioResult / metadata
│   ├── adapters.py         # NumPy/Torch/路径/mapping adapter registry
│   └── errors.py           # 稳定公开异常
├── preprocess_bwe.py       # 48 kHz 母带 → BWE 配对数据
├── synthesize.py           # 统一特征/音频 CLI
├── infer.py                # 兼容旧 NPY CLI，内部调用 SDK
├── export.py               # 纯 checkpoint/TorchScript/ONNX
└── benchmark.py            # dtype、RTF 与峰值内存基准
```

核心模型 `NHNVocoder.forward(features)` 不接受路径、WAV 或采样率参数。所有外部输入在
进入 forward 前完成适配和验证，这能让模型导出与服务部署保持简单。

## CLI 兼容与扩展

保留现有命令：

```bash
uv run nhn-infer input.npy best.pt output.wav --device cpu
```

已增加上层友好的统一入口：

```bash
# 自动读取任意 <=48 kHz WAV，并固定输出 48 kHz
uv run nhn-synthesize input.wav vocoder_bwe.pt output_48k.wav --device cpu

# 显式数组格式
uv run nhn-synthesize features.npz vocoder.pt output.wav \
  --input-format llsm72

# 执行完整推理诊断并输出 metadata，但不写 WAV 文件
uv run nhn-synthesize input.wav vocoder_bwe.pt output.wav --validate-only
```

CLI 可以读取 WAV header 的原始采样率，但不能猜测裸 72 维数组的来源带宽；数组输入
的这项语义必须来自 feature metadata。

## 评测方案

每个主要输入采样率至少建立以下四类固定测试：纯净歌声、气声/清辅音、高音与滑音、
含轻微真实噪声的低采样率来源。指标分开报告：

| 范围 | 指标 | 目的 |
| --- | --- | --- |
| 全频 | MR-STFT、Mel L1 | 总体重建 |
| 0–源 Nyquist | 回降采样 SI-SDR/L1 | 已知频带是否被破坏 |
| 源 Nyquist–24 kHz | log-STFT、能量误差 | 缺失高频补全程度 |
| F0/谐波 | 谐波频点误差 | 高音与音色稳定性 |
| 性能 | RTF、峰值内存 | CPU/GPU 可部署性 |

客观指标不能替代盲听。盲听至少比较：低采样率直接上采样、统一 vocoder+BWE、真实
48 kHz target；样本顺序随机，重点记录齿音、气声、金属感、固定嘶声和瞬态模糊。

## 建议验收门槛

- 输出 WAV header 必须为 48,000 Hz、mono，样点数符合帧数规则。
- 输出回降到原始输入采样率后不得明显劣于输入低带宽音频。
- 源 Nyquist 以上至 24 kHz 的能量不能塌缩为零，也不能在静音段持续产生固定噪声。
- 相同输入、checkpoint 和 seed 可复现。
- 两秒输入的 CPU 推理继续低于两秒，并单独记录分析、重采样和声码器耗时。
- 训练/验证清单无同歌曲泄漏，所有对齐误差在允许范围内。
- 至少完成一次固定样本 ABX/盲听后，才将 BWE 标记为可用。

## 长音频与流式预留

第一版先实现整段和 batch。第二版增加：

```python
session.synthesize_chunked(features, chunk_frames=..., overlap_frames=...)
session.stream(feature_chunks, lookahead_frames=...) -> Iterator[AudioChunk]
```

动态 FIR、非因果卷积和 PQMF 都需要上下文，所以不能直接把帧硬切开。分块接口必须：

- 左右保留 receptive-field 上下文；
- 固定随机噪声在全局时间轴上的位置；
- 对有效区做等功率 crossfade；
- 返回 `start_sample/end_sample/is_final`；
- 用整段推理结果做边界误差回归测试。

## 错误模型与版本

公开稳定异常类型：`InvalidFeatureShape`、`FeatureValueError`、
`FeatureConfigMismatch`、`CheckpointCompatibilityError`、`UnsupportedInputType`。

72 维布局命名为 `llsm72-v1`；checkpoint 已有独立 `format_version`。两者不能混用：
前者描述特征语义，后者描述权重/网络兼容性。SDK 公共 API 按语义版本管理。

## 实施状态

### SDK 第一阶段（已完成）

- 建立 `LLSMFeatures`、`AudioResult`、校验器和 `VocoderSession`。
- 支持 NumPy、Torch、路径和字段 mapping。
- 让现有 `nhn-infer` 改为调用 SDK，但保持原 CLI 参数兼容。
- 测试布局转换、错误输入、seed 复现、checkpoint 采样率和 batch。

### SDK 第二阶段（已完成）

- 加入 WAV/波形分析适配器，复用 FCPE/RMVPE/pyllsm2 实例。
- 加入统一 `vocoder_bwe` 模式、原始采样率 metadata 和训练覆盖范围校验。
- 新增从 48 kHz 母带生成多种低采样率条件的统一预处理入口。
- 加入插件式 adapter registry，供 MeloStudio 上层注册声学模型输出。
- 加入设备、线程数和推理统计信息。

### SDK 第三阶段（基础部署已完成）

- 重叠分块长音频推理。
- 纯推理 checkpoint、TorchScript/ONNX 后端适配。

尚未实现的服务层能力：真正的增量流式迭代器和取消机制。现有
`synthesize_chunked` 面向长音频离线推理，使用重叠交叉淡化并严格保持目标长度；它
不是低延迟流式协议。

## 验收标准

- 原有 `.npy -> WAV` 命令输出长度与 SDK 完全一致。
- `[T,72]`、`[72,T]`、batch 和字段 mapping 结果一致。
- 相同 checkpoint、输入和 seed 逐样本可复现。
- WAV header 的采样率始终等于 checkpoint 配置。
- 每一种低采样率分支都与 48 kHz target 的起点、时长和 LLSM 帧数严格对齐。
- 输入采样率超出 checkpoint 已训练范围时给出明确 warning 或在严格模式拒绝。
- 模型实例只加载一次，连续推理不会重复加载权重。
- 上层新增一种输入只需实现 adapter，不需要修改 `NHNVocoder.forward`。

# P5：DSP 控制层与上层后训练方案

## 当前定位

P5 第一版用于验证“稳定 NHN 基模 + SDK 控制协议 + 小型后训练层”的路线。基模仍然只
接收 LLSM72，并保持原有 48 kHz 输出契约；DSP 不扩展核心 `forward` 的 72 维输入，
因此上层模型和基模可以独立演进。

```text
上层声学模型 / LLSM72
          ↓
SynthesisRequest(features, DSPControl)
          ↓
冻结的 NHN Base Vocoder
          ↓
可微 DSP（帧级控制、采样级处理）
          ↓
48 kHz mono WAV
```

## 六维 DSP 控制

| 控制 | 范围 | 作用 | 零值行为 |
| --- | ---: | --- | --- |
| `gain_db` | -18–18 dB | 帧级响度包络 | 不改变增益 |
| `harmonic_tilt` | -1–1 | 一阶高频/谐波倾斜 | 不改变频谱 |
| `breathiness` | 0–1 | 按局部能量加入气声噪声 | 不加噪声 |
| `transient_gain` | -1–1 | 增强或抑制快速变化 | 不改变瞬态 |
| `deesser_amount` | 0–1 | 轻量高频平滑 | 不去齿音 |
| `limiter_amount` | 0–1 | 可变软限幅 | 不限幅 |

控制可以是标量，也可以是长度与 LLSM 帧数相同的一维数组。全零控制是精确旁路。当前
实现是用于验证控制、训练稳定性和 CPU 成本的轻量 DSP，不应在完成真实数据消融和
盲听前将它描述为成熟母带处理器。

## 上层 SDK

```python
import numpy as np
from vocoder import DSPControl, SynthesisRequest, VocoderSession

session = VocoderSession.from_checkpoint("checkpoints/base/best.pt", device="cpu")
controls = DSPControl(
    gain_db=-1.5,
    harmonic_tilt=np.linspace(-0.1, 0.2, len(llsm72)),
    breathiness=0.08,
    deesser_amount=0.15,
    limiter_amount=0.1,
)
request = SynthesisRequest(
    features=llsm72,
    dsp=controls,
    metadata={"speaker_id": "singer-01", "style": "soft"},
)
result = session.synthesize_request(
    request, chunk_frames=750, overlap_frames=32, seed=1234
)
result.save("output_48k.wav")
```

以后上层只需实现已有 `FeatureAdapter`，将自己的输出整理成 LLSM72，再选择直接给出
`DSPControl`，或者使用后训练 checkpoint 内置的自动控制头。`request.metadata` 不送入
网络，仅用于上层追踪；speaker/style embedding 的可训练接口等确定上层张量尺寸后再加。

## 冻结基模后训练

后训练直接复用 P1/P3 的配对数据，不需要重新预处理：

```bash
uv run nhn-train --config configs/nhn-dsp-post.yaml
```

两个阶段使用不同配置，文件头分别声明用途和 `config_type`，随后执行：

```bash
uv run nhn-train --config configs/nhn-base.yaml
uv run nhn-train --config configs/nhn-dsp-post.yaml
```

CLI 显式参数会覆盖 YAML；例如显存不足时追加 `--batch-size 1`。

训练期间：

- NHN 基模始终 `eval()` 且 `requires_grad=False`；
- 只训练 17,414 参数（float32 约 0.066 MiB）的轻量 `DSPControlPredictor`；
- 重建损失继续使用 waveform/STFT/Mel/phase/IF/harmonic 组合；
- 加入控制幅度正则，防止控制头用过强 DSP 掩盖基模问题；
- 输出 checkpoint 同时保存基模、DSP 控制头、DSP 配置和控制头 optimizer。

加载后无需改变调用代码：

```python
session = VocoderSession.from_checkpoint("checkpoints/nhn-dsp/latest.pt")
result = session.synthesize(llsm72)  # 自动使用预测的 DSP 控制
```

传入显式 `DSPControl` 时，显式控制优先于 checkpoint 的自动控制头。

## 实验与后续边界

第一轮至少比较三组：基模旁路、基模加手工控制、基模加自动控制头。重点听齿音、气声、
高音金属感、爆破音和响度抽动，同时比较 Mel/STFT、RTF 和峰值内存。

当前纯推理 `.pt` 导出会保留 DSP 控制头；TorchScript/ONNX 暂时明确拒绝 DSP 后训练
checkpoint，防止静默导出成不带 DSP 的基模。确认音质收益后再将控制头和 DSP 算子合并
到部署图。真正的上层声学模型后训练还需确定其输出张量、speaker/style 条件和 teacher
数据格式；P5 第一版先保持该接口开放，不虚构固定维度。

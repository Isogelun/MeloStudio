# NHN vocoder 完善路线

## 当前定位

当前实现是一个可训练的 NHN-inspired 原型：输入 pyllsm2 Coder 产生的 72 维
LLSM 特征，输出 48 kHz 波形。参考模型只有结构截图；缺失的 forward 已按可解释的
动态 FIR/source-filter 方式自主实现，因此不宣称是原项目的逐算子复刻。

## 第一优先级：训练基础设施（已完成）

- [x] Python 3.12 的项目内 uv 环境
- [x] `pyproject.toml`、`uv.lock` 与命令行入口
- [x] 训练/分析/RMVPE/开发依赖分组
- [x] 固定长度随机裁剪、补齐和 batch 训练
- [x] 固定验证集划分或独立验证目录
- [x] CUDA AMP、梯度裁剪与 cosine 学习率调度
- [x] latest/best/周期 checkpoint
- [x] model、optimizer、scheduler、scaler、epoch 和 step 恢复
- [x] 控制台、JSONL 和 TensorBoard 日志
- [x] 72 维特征 mean/std/min/max/p01/p99 统计
- [x] 使用 p01/p99 稳健范围归一化
- [x] 归一化 buffer 随模型 checkpoint 保存
- [x] 一条命令完成 WAV 复制、F0、LLSM 72 维编码、统计和报告

## 第二优先级：音质

- [x] 用 Kaiser 原型与余弦调制分析/合成滤波器的真实 PQMF 替换 polyphase 拆分
- [x] 将 5 维 BAP 插值映射到十路 Mel 间隔带限噪声，不再使用全频平均增益
- [x] 按截图通道关系自主实现 SpectrumGenerator/FIRFilter forward，并明确标为 NHN-inspired
- [x] 增加 Mel、圆周相位、瞬时频率和 F0 谐波加权损失
- [x] 增加训练期轻量 MPD/MSD、LSGAN 目标与 feature matching

### 第二优先级的设计取舍

本轮没有直接替换成完整 NSF-HiFiGAN 生成器。当前生成器已经把 F0 构造出的谐波
激励、BAP 控制的噪声激励送入非因果滤波网络，结构上属于 source-filter 路线；继续
保留它可以维持约 12 MB 的推理权重。借鉴 HiFi-GAN 的部分放在训练端：周期判别器
负责基频及谐波周期，尺度判别器负责不同时域结构，feature matching 稳定生成器。
这些判别器不会进入推理 checkpoint 的模型对象，也不会增加部署参数量。

重新核对截图后，三个 SpectrumGenerator head 均按 `256 -> 256` 输出解释，每个 head
为对应的谐波、噪声、瞬态激励预测一条每帧 256-tap FIR。FIR 使用 512 点 FFT 做线性
卷积，并通过 overlap-add 拼接相邻帧。这取代了早期临时使用的
`direct/gain/gate` 解释：新的 forward 与截图中的“三个 head + 三个无参数 FIRFilter”
一一对应，也具有明确的动态声道滤波含义。由于没有原始源码，这仍是自主实现，而不是
已证实与原作者逐算子一致。

新 checkpoint 标记为 format v3。v1/v2 的旧 head 输出是 `hop*3`，加载时迁移器会
保留第一组 `hop` 权重作为 FIR 初始化；同时 secondary WaveNet 从有混叠相位通道改为
真实 PQMF 子带。旧 optimizer 状态因参数形状变化不会恢复；旧权重必须继续微调，
不能期待直接推理保持原音质。

训练默认先进行 10,000 step 的纯重建预热，再开启 GAN。可通过
`--gan-start-step` 调整；小数据过拟合实验可设为 100–1000，但不建议从第 0 step
直接对抗训练。生成器总损失为：

```text
L_G = L_wave/STFT/Mel/phase/IF/harmonic + 1.0 * L_adv + 2.0 * L_fm
```

参考依据：HiFi-GAN 官方实现使用周期集合 2/3/5/7/11、multi-scale
discriminator、LSGAN 和 feature matching；SiFiGAN/Source-Filter HiFi-GAN 则说明
显式周期激励与分层滤波尤其适合歌声。PQMF 实现沿用 ParallelWaveGAN 的
Kaiser-window cosine-modulated 设计，并针对 16 子带采用更窄截止频率和 254 taps。

- HiFi-GAN: <https://github.com/jik876/hifi-gan>
- HiFi-GAN paper: <https://arxiv.org/abs/2010.05646>
- SiFiGAN paper: <https://arxiv.org/abs/2210.15533>
- ParallelWaveGAN PQMF: <https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/layers/pqmf.py>

### 音质验收

代码级检查只能证明长度、梯度和数值稳定，不能证明主观音质变好。需要在相同数据、
训练步数与验证片段下至少跑 baseline 与本版本两组，记录 Mel/STFT 损失，并做盲听。
尤其需要观察清辅音、高音长音、气声、音高滑音，以及片段边界是否出现相位点击。

## 第三优先级：数据分析（已完成）

- [x] 在批量任务中复用 FCPE/RMVPE 模型实例
- [x] 自动重采样、float/24-bit WAV 和 FLAC
- [x] 按 48 kHz 母带构造严格对齐的多采样率→48 kHz BWE 训练对
- [x] 增加 BWE manifest、重采样配置与对齐质量报告
- [x] 削波、静音、F0 跳变、NaN/Inf 和长度质量报告
- [x] FCPE/RMVPE 数据集级对照报告

### P3 已实现入口

```bash
# 普通数据：自动读取 WAV/FLAC 和任意原始采样率，统一生成 48 kHz 配对数据
uv run nhn-preprocess raw_audio data/train --f0-backend fcpe

# 48 kHz 母带：生成多种来源带宽的 LLSM72，target 始终为原始 48 kHz
uv run nhn-preprocess-bwe masters_48k data/bwe_train \
  --source-sample-rates 8000,12000,16000,22050,24000,32000,44100,48000

# 数据集级 F0 后端对照
uv run nhn-compare-f0 raw_audio reports/f0-comparison.json \
  --first fcpe --second rmvpe
```

BWE 输出采用 `features/name@采样率.npy`、共享的 `targets/name.wav` 和
`manifests/train.jsonl|valid.jsonl`。训练器检测到 manifest 后会按母带 id 分组：训练时
每个母带随机抽一个采样率变体，验证时保留全部变体。

## 第四优先级：部署

- [ ] 按 [输入扩展与 SDK 规划](vocoder-input-sdk-plan.md) 实现稳定上层接口
- [ ] 统一 `vocoder_bwe` checkpoint 的输入采样率 metadata 与覆盖范围校验
- [ ] 重叠分块长音频推理
- [ ] 纯推理 checkpoint 导出
- [ ] TorchScript/ONNX 导出
- [ ] float16/bfloat16/int8 基准
- [ ] CPU/GPU 的 RTF 与峰值内存基准

## 推荐验收顺序

1. 用 5–10 条音频进行过拟合测试。
2. 确认训练、验证损失和重建音频均正常。
3. 用完整数据训练 baseline。
4. 再逐项实现第二优先级，每项都做同一验证集的消融对比。

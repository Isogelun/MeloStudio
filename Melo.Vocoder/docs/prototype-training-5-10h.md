# 5～10 小时数据原型训练指南

这套方案用于验证 NHN Vocoder 是否能够学到可辨识音色、正常进入 GAN 阶段，并完成
checkpoint、推理和导出链路。它比冒烟测试更有意义，但仍不等同于正式成品训练。

对应配置：[nhn-prototype-5-10h.yaml](../configs/nhn-prototype-5-10h.yaml)。
NVIDIA 训练机应先按 [README 的 PyTorch CPU 与 CUDA 版本说明](../README.md#pytorch-cpu-与-cuda-版本)
单独安装 CUDA wheel；下面命令使用 `--no-sync`，避免重新同步成默认 CPU wheel。

## 数据建议

- 使用单个歌手约 5～10 小时清洗后的有效干声。
- 母带使用 48 kHz，尽量无混响、无削波、无伴奏泄漏。
- 建议按歌曲或录音 session 划分测试集，再把训练部分切成约 5～12 秒的文件。
- 至少保留 20～30 分钟验证数据，并额外保留约 20～30 分钟完全不参与训练的试听测试集。
- 音域、力度、气声、清辅音和转音覆盖比重复增加相似句子更重要。

## 预处理

普通 LLSM72 → 48 kHz 声码器：

```bash
cd /Users/ad/MineCode/MeloStudio/vocoder

uv run --no-sync nhn-vocoder preprocess raw_wavs data/prototype_5_10h \
  --f0-backend fcpe --f0-device cuda
```

如果本次验证目标是低采样率输入、固定输出 48 kHz，则使用 BWE 预处理：

```bash
uv run --no-sync nhn-vocoder preprocess --pipeline bwe masters_48k data/prototype_5_10h \
  --source-sample-rates 16000,24000,32000,48000 \
  --f0-backend fcpe --f0-device cuda
```

原型阶段先使用四种来源采样率，可以减少特征提取和磁盘成本；完整训练再扩展到更多档位。

## 训练

```bash
uv run --no-sync nhn-vocoder train --config configs/nhn-prototype-5-10h.yaml
```

配置默认行为：

- `batch_size: 1`、`segment_seconds: 1.0`；
- 训练 12 epoch；
- 前 5000 step 只训练重建损失；
- 5000 step 后启用 MPD/MSD、LSGAN 和 feature matching；
- GAN 权重先用 0.5，降低首次原型训练崩坏风险；
- 每 500 step 更新一次 `latest.pt`，验证最优模型保存为 `best.pt`。

训练产物位于：

```text
checkpoints/nhn-prototype-5-10h/
├── latest.pt
├── best.pt
├── training.jsonl
└── tensorboard/
```

断点恢复时，将配置中的 `resume` 改成：

```yaml
resume: ../checkpoints/nhn-prototype-5-10h/latest.pt
```

## 为什么文件数量会影响训练时长

当前数据集每个文件在每个 epoch 随机抽取一个 `segment_seconds` 片段。训练 step 数近似为：

```text
每 epoch step = ceil(训练文件数 / batch_size)
总 step = 每 epoch step × epochs
```

假设平均每个文件 8 秒：

| 有效数据 | 文件数估算 | 扣除 5% 验证后 | 12 epoch 总 step |
| --- | ---: | ---: | ---: |
| 5 小时 | 约 2250 | 约 2138 | 约 25,700 |
| 10 小时 | 约 4500 | 约 4275 | 约 51,300 |

推荐原型目标约 25,000～50,000 step。如果实际文件明显更长、文件数很少，应按以下公式增加
epoch：

```text
epochs = 目标总 step × batch_size / 训练文件数
```

例如只有 1000 个训练文件，目标 30,000 step、batch=1，则约需 30 epoch，而不是默认 12。

## 训练时间预估

以下时间包含重建阶段和更慢的 GAN 阶段，是当前实现尚未在你的具体机器上跑正式数据前的
保守估算：

| 机器 | 推荐设置 | 约 2.5万 step | 约 5万 step |
| --- | --- | ---: | ---: |
| 12～16 GB CUDA GPU | batch=1，0.5～1 秒 | 18～40 小时 | 36～80 小时 |
| 24～32 GB CUDA GPU | batch=1，1 秒 | 10～24 小时 | 20～48 小时 |
| 48～96 GB 高端 CUDA GPU | batch=1～2，1～2 秒 | 7～18 小时 | 14～36 小时 |
| Apple MPS | batch=1，0.5～1 秒 | 1～4 天 | 2～8 天 |
| 纯 CPU | 仅建议排错 | 数天以上 | 不建议完整运行 |

因此，如果你使用一张 24～32 GB 的近期 NVIDIA GPU，5 小时数据预计约 10～24 小时，
10 小时数据预计约 20～48 小时。动态 FIR、STFT 损失和 GAN 判别器的具体算子效率会让不同
GPU 偏离这个范围；模型约 12 MB 并不代表训练计算量小。

预处理另计：使用 GPU FCPE 通常应预留约 1～4 小时；CPU FCPE、RMVPE 或大量 BWE 采样率
变体可能需要更久。

## 如何得到你机器上的准确时间

先运行训练并观察进入稳定阶段后的实际 step 速度。分别记录 GAN 启动前后各 200～500 step
所需时间，然后估算：

```text
总小时 ≈ (5000 × 重建阶段秒/step
          + (目标总 step - 5000) × GAN 阶段秒/step) / 3600
```

不要只用 GAN 启动前的速度推算全程；GAN 开启后需要额外训练 MPD/MSD，单 step 会明显变慢。

## 试听与停止条件

建议至少试听以下 checkpoint：GAN 启动前、GAN 启动后约 2000 step、`best.pt` 和最终
`latest.pt`。重点检查：

- 音高和节奏是否正确；
- 清辅音和气声是否变成宽带噪声；
- 高频是否出现周期性金属音；
- GAN 启动后是否突然爆音、音量漂移或音色崩坏；
- 16 kHz 输入能否稳定输出 48 kHz，且已知低频带没有明显受损。

如果 GAN 启动后明显变差，可将 `adversarial_weight` 从 0.5 降至 0.25，或把
`gan_start_step` 从 5000 推迟到 10000。若验证损失连续数个 epoch 不再改善且试听也无提升，
可以提前停止，不必机械跑满 12 epoch。

更多数据规模和机器建议见[推荐数据量与训练机器](training-data-and-hardware.md)，完整命令见
[项目 README](../README.md)。

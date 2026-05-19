# 训练日志

## Run #1 — 2026-05-19

### 配置

- 基座：`Qwen/Qwen2.5-1.5B-Instruct`（本地，ModelScope 下载）
- 数据：142 train / 10 valid / 5 test（chat messages 格式）
- LoRA：rank=8, scale=20, dropout=0, num_layers=8
- 训练：batch_size=2, iters=600, lr=1e-5, max_seq_length=1024
- 硬件：MacBook Air M5（16GB / 10-core GPU）
- 可训练参数：**2.638M / 1543.714M ≈ 0.171%**
- 训练时长：约 1.5 分钟（峰值显存 3.95GB）

### Loss 曲线

| Iter | Train loss | Val loss | 备注 |
|---:|---:|---:|---|
| 1   | —     | **3.148** | 起点 |
| 100 | 0.493 | **0.629** | ⭐ Val 最低点 |
| 200 | 0.343 | 0.679 | |
| 300 | 0.140 | 0.733 | |
| 400 | 0.114 | 0.784 | |
| 500 | 0.096 | 0.804 | |
| 600 | 0.084 | **0.840** | 最终（已过拟合）|

### 诊断：典型过拟合

- Train loss 从 1.418 → 0.084，跌了 **17×**。
- Val loss 从 iter 100 的 0.629 单调上升到 0.840。
- 训练 loss 还在降、验证 loss 反而升 → 模型在**背训练样本**，泛化能力下降。
- 根因：142 条样本 × batch 2 = 71 步/epoch；600 iters ≈ **8.5 个 epoch**，对小数据集太多。

### 应对：切回 val 最低的 checkpoint

mlx-lm 每 200 步保存一次，目录里有：
- `adapters/0000200_adapters.safetensors` ← 离最佳点最近（val 0.679）
- `adapters/0000400_adapters.safetensors`
- `adapters/0000600_adapters.safetensors` ← 当前 `adapters.safetensors`

切换最佳 checkpoint：
```bash
cp adapters/0000200_adapters.safetensors adapters/adapters.safetensors
bash scripts/generate.sh "请将以下白话文翻译为文言文：今天加班到很晚。"
```

### 其他观察

- **显存极度富裕**：3.95GB / 16GB。完全可以上 batch_size=4 或换 3B 基座。
- **训练超快**：600 步约 90 秒（~7 it/s）。下一轮可以多跑几组超参对比。

### 测试集评估（iter 600 final adapter）

```
Test loss 0.858, Test ppl 2.359
```

- Test loss (0.858) ≈ Val loss (0.840)，两者一致 → 测试集和验证集分布相近，**数字真实反映过拟合后的泛化水平**。
- Perplexity 2.36 含义：模型对正确答案的平均"分支预测数"约 2.4 个 token，对生成任务属于**可用但不惊艳**的水平（理想 < 2.0）。
- 用 iter 200 的 adapter 重测，理论上 test loss 应该降到 ~0.68 附近——见下方对照。

### Checkpoint 对照（同一测试集 + 同一 prompt）

| 指标 | iter 200 (val 最低) | iter 600 (final) |
|---|---|---|
| Test loss | **0.640** ✅ | 0.858 |
| Test ppl  | **1.896** ✅ | 2.359 |
| Peak mem  | 3.205 GB | 3.218 GB |
| 输出 tokens | 18 | 23 |

Prompt：`请将以下白话文翻译为文言文：今天加班到很晚，地铁都停运了，只能打车回家。`

- **iter 200 输出**：`今日夜深方归，其时地铁已歇，唯有雇车而已。`
  - 简洁、地道——"其时"、"而已" 是真正的文言虚词。
- **iter 600 输出**：`今日值守至夜深，地铁已尽停载，孤身一人，唯有雇车归家。`
  - 多出的"孤身一人"是训练数据里没要求的修饰，**过拟合后模型在套训练集里的修辞模板**。

困惑度 2.36 → 1.90 是约 20% 的提升，且 1.90 已踏进"流畅"门槛（< 2.0）。**结论：早停（iter 200）的 checkpoint 才是 Run #1 真正的产物**。

> 操作记录：
> ```bash
> cp adapters/adapters.safetensors adapters/_final_iter600.safetensors  # 备份
> cp adapters/0000200_adapters.safetensors adapters/adapters.safetensors  # 切到最佳
> ```

---

## 下一轮（Run #2）计划

### 方案 A — 仅调超参（治标）

```yaml
iters: 200            # 停在 val 拐点附近
save_every: 50        # 更密的 checkpoint
steps_per_eval: 25    # 更早发现过拟合
lora_parameters:
  rank: 4             # 容量减半
  scale: 16.0
  dropout: 0.05       # 加正则
```

### 方案 B — 扩数据（治本，推荐）

142 条对 1.5B 模型太少。让大模型（GPT-4 / Claude / Qwen-Max）批量生成 300–500 条"现代场景文言化"对照，混进 train.jsonl，再用方案 A 的超参跑。

### 方案 C — 换基座

显存还剩 12GB，可换 `Qwen2.5-3B-Instruct`，batch_size 降到 1，看是否仅靠更大的基座就缓解过拟合。

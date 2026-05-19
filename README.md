# qwen-wenyan-mlx

在 Apple Silicon (M-series) 上用 **MLX-LM** 对 **Qwen2.5-1.5B-Instruct** 做 LoRA 微调，得到一个**白话 ↔ 文言文互译**的小模型。

> 硬件参考：MacBook Air M5，16GB RAM / 10-core GPU。1.5B + LoRA 全程显存峰值约 6–8GB，训练 600 步约 15–25 分钟。

---

## 📂 目录结构

```
.
├── data/                  # 数据集（chat messages 格式，mlx-lm 原生支持）
│   ├── train.jsonl        # 142 条
│   ├── valid.jsonl        # 10 条
│   └── test.jsonl         # 5 条
├── lora_config.yaml       # LoRA 训练配置
├── scripts/
│   ├── train.sh           # 训练
│   ├── eval.sh            # 测试集 loss
│   ├── generate.sh        # 单条推理
│   ├── fuse.sh            # 合并 adapter 到基座
│   └── chat.py            # 交互 REPL
└── README.md
```

### 数据集说明

每行一个样本，OpenAI chat 格式：

```json
{"messages": [
  {"role": "system", "content": "你是一个精通文言文的翻译助手……"},
  {"role": "user",   "content": "请将以下白话文翻译为文言文：……"},
  {"role": "assistant", "content": "……"}
]}
```

内容覆盖：
- **经典古籍（约 50 条）**：《论语》《孟子》《道德经》《孙子兵法》《出师表》《兰亭集序》《赤壁赋》
- **双向翻译（约 15 条）**：白话↔文言两个方向都有
- **现代场景文言化（约 80 条）**：手机、咖啡、地铁、加班、网购、AI、电动车——本数据集的精髓，让模型学会处理训练语料里见不到的现代概念

---

## 🚀 完整流程

### 0. 环境准备（已完成）

```bash
python -m venv mlx-env
source mlx-env/bin/activate
pip install mlx-lm datasets
```

> mlx-lm 会在首次运行时自动从 HuggingFace 拉取基座模型（约 3GB），请保持网络畅通。如在国内，可设置 `export HF_ENDPOINT=https://hf-mirror.com`。

### 1. 训练

```bash
source mlx-env/bin/activate
bash scripts/train.sh
```

等价于：

```bash
mlx_lm.lora --config lora_config.yaml
```

训练过程中会看到：
- 每 20 步打印一次训练 loss
- 每 100 步在 valid.jsonl 上算一次验证 loss（关注它是否还在下降）
- 每 200 步保存一次 adapter 到 `adapters/`

**判断收敛**：train loss 通常从 ~2.5 降到 ~0.8 左右；valid loss 若反弹，说明过拟合，可减小 `iters` 或 `lora rank`。

### 2. 测试集评估

```bash
bash scripts/eval.sh
```

输出 test.jsonl 上的平均 loss / perplexity。

### 3. 单条推理（最直观）

```bash
bash scripts/generate.sh "请将以下白话文翻译为文言文：今天我加班到很晚，地铁都停运了，只能打车回家。"
```

或自定义 prompt：

```bash
bash scripts/generate.sh "请将以下文言文翻译为白话文：吾日三省吾身。"
```

### 4. 交互 REPL

```bash
python scripts/chat.py
```

```
你> 请将以下白话文翻译为文言文：人工智能正在改变世界。
模型> 智机渐易天下也。
```

### 5. 合并 adapter → 独立模型（可选）

```bash
bash scripts/fuse.sh
# 产物在 fused-model/，可直接用 mlx_lm.generate --model fused-model 加载
```

---

## ⚙️ 关键超参（lora_config.yaml）

| 参数 | 值 | 说明 |
|---|---|---|
| `num_layers` | 8 | 仅微调顶部 8 层（共 28 层），省显存 |
| `batch_size` | 2 | 16GB 安全值；若 OOM 改为 1 |
| `iters` | 600 | 142 条样本 × batch 2 ≈ 8 个 epoch |
| `learning_rate` | 1e-5 | LoRA 经验值 |
| `max_seq_length` | 1024 | 文言句子普遍较短，1024 够用 |
| `lora.rank` | 8 | rank 越大表达力越强、过拟合风险越高 |
| `lora.scale` | 20.0 | 即 α=20，等效 α/rank=2.5 |

---

## 🔧 常见问题

**OOM / 内存吃紧** → `batch_size: 1`、`num_layers: 4`、`max_seq_length: 512`
**Loss 不降** → 检查数据格式；提高 `learning_rate` 到 `2e-5`
**输出重复 / 啰嗦** → 推理时 `--temp 0.3` 降到 `0.1`，或加 `--repetition-penalty 1.1`
**想换 3B 基座** → `model: "Qwen/Qwen2.5-3B-Instruct"`，并把 `batch_size` 降到 1

---

## 📚 参考

- [MLX-LM 仓库](https://github.com/ml-explore/mlx-lm)
- [Qwen2.5 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

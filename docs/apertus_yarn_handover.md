# EAGLE3-Apertus YaRN 长上下文微调 · 交接文档

> 2026-10-04 · 代码：`TanZhendong/SpecForge-Yarn` @ `39291b0`（main）

## 1. 背景与目标

- 组合：target `swiss-ai/Apertus-8B-Instruct-2509`（8B，64K ctx，xIELU，需 transformers≥4.56）+ draft `thomaskiefer/EAGLE3-Apertus-8B-Instruct-2509`（513M 单层 EAGLE3，SpecForge 训练，**训练长度仅 4096**）。
- 问题：draft 的 rope（theta=10000，无缩放）超出 4k 位置即失效——PG-19 上 accept 从 ~1.3 崩到 0.6。
- 已做：零训练 YaRN 转换（factor 16 / orig 4096 / max 64k）已救回长上下文（10k accept 2.05、30k 2.31）。
- 本次训练：用 PG-19 长文本在 **YaRN rope 生效状态下继续微调**，目标把长上下文 accept 再抬一档（llama3 同配方先例）。

## 2. 训练方式

`scripts/train_eagle3_online_yarn.py`（SpecForge-Yarn fork 自带）。"online" 指 target 隐状态每 batch 现场算（冻结 teacher-forcing 前向，**无自回归生成**）；不用 offline 路径（64k 数据预计算隐状态需 ~10TB）。

## 3. 三个必须知道的对齐约定

1. **隐状态层号 {2, 16, 29}**（input-to-layer，32 层 target）。SpecPV 推理端、sglang 路径默认均为 {2,16,29}；但 online_yarn 脚本默认是 {2,17,29}（mid 错一层）。已在 `configs/apertus-8B-eagle3-yarn.json` 里用 `eagle_config.eagle_aux_hidden_state_layer_ids=[1,15,28]` 钉死，勿删。
2. **rope_scaling 双 key**：config 同时写 `type` 和 `rope_type`（SpecPV 读前者、SpecForge 读后者），训完的 checkpoint 两边通用。
3. **词表映射**：resume 微调时脚本自动跳过 d2t/t2d 重建（保留 author checkpoint 自带的 32000 子词表），此逻辑已在脚本里，勿动。

## 4. 配方（与 llama3-yarn 完全一致）

| 参数 | 值 |
|---|---|
| max-length / LR / epochs / warmup | 32768 / 2e-5 / 1 / 0.05 |
| ttt-length / attention 后端 | 4 / flex_attention |
| global batch | 16（micro=1；accumulation 自动= 16÷卡数） |
| 有效训练量 | 400 optimizer steps ≈ 2.1 亿 token |
| 耗时参考 | 2×A100 ≈ 45h；**4×A100 ≈ 22h** |

## 5. 新服务器步骤

```bash
git clone https://github.com/TanZhendong/SpecForge-Yarn.git && cd SpecForge-Yarn
bash setup_env.sh          # uv 环境；注意：transformers 用 4.57.6（requirements 里的 4.55.2 不支持 Apertus），不装 sglang

# 数据（1.08GB，走镜像）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download hcyy/pg19-yarn-6400 \
  --repo-type dataset --local-dir ~/datasets/pg19-yarn-6400
.venv/bin/python scripts/prepare_pg19_data.py \
  --src ~/datasets/pg19-yarn-6400/pg19_yarn_6400_64k.parquet \
  --dst cache/dataset/train_pg19_64k_6400.json

# 模型：Apertus-8B（16GB）与 author draft 各放到 models/ 下（路径按下面改脚本或建软链）
# 起点 checkpoint：author draft 权重 + yarn config；不要拷 training_state.pt！
mkdir -p outputs/apertus-eagle3-yarn/epoch_9
cp configs/apertus-8B-eagle3-yarn.json outputs/apertus-eagle3-yarn/epoch_9/config.json
ln -s <models>/EAGLE3-Apertus-8B-Instruct-2509/model.safetensors outputs/apertus-eagle3-yarn/epoch_9/

# 检查 run_apertus_eagle3_yarn.sh 里 --target-model-path 后开训（tmux）
bash examples/run_apertus_eagle3_yarn.sh 4
```

## 6. 产物验证（SpecPV，训练服务器或拷回本机）

```bash
EAGLE_MODEL_PATH=<outputs>/apertus-eagle3-yarn/epoch_0 \
  python tests/diag_apertus_ctx_sweep.py        # 512–8k 扫描
EAGLE_MODEL_PATH=<outputs>/apertus-eagle3-yarn/epoch_0 \
  python tests/test_apertus_longcontext.py      # 10k / 30k
```

对比基线（PG-19 单样本，greedy，full verify）：

| draft | ≤4k | 10k | 30k |
|---|---|---|---|
| 原始（无 yarn） | 1.15–1.35 | 0.62 | 0.62 |
| yarn 零训练（`~/models/EAGLE3-Apertus-8B-Instruct-2509-yarn64k`） | 0.99–1.45 | 2.05 | 2.31 |
| **本次微调目标** | ≥1.3 | >2.5 | >2.8 |

## 7. 坑位清单

- `epoch_9` 目录里混入 `training_state.pt` → resume 继承 epoch=9，`--num-epochs 1` 会一步不训。
- 产物目录名是 **epoch_0**（代码按 `epoch % save_interval` 存）；中断重跑时 `get_last_checkpoint` 取数字最大的目录，若 epoch_9 还在会回到起点——续训前把起点改名 `epoch_init`。
- transformers <4.56 装不上/跑不了 Apertus；requirements.txt 里钉的 4.55.2 是错的，setup_env.sh 已覆盖。
- 显存：每卡 = 16GB 冻结 target + 32k 激活 + 全词表 logits 峰值（~8.6GB）。2 卡 A100-80G 时 GPU1 会贴到 ~81GB；4 卡无此压力。
- wandb 默认关闭（`--report-to none`）；要开改 `--report-to wandb` + `export WANDB_MODE=offline`（免 key）。
- 微调后短上下文聊天域 accept 会从 5.73 回落（PG-19 域偏移，正常现象）。

## 8. 本机遗留状态（tanzhendong 机器）

- commit `39291b0` **未 push**（`git push origin main` 即可）。
- tmux `apertus_yarn` 里的 2 卡训练（45h 档）还在跑，弃用则 `tmux kill-session -t apertus_yarn`。
- 路径速查：target/draft 在 `~/models/`，原始数据 `~/datasets/pg19-yarn-6400/`，训练 jsonl `~/SpecForge-Yarn/cache/dataset/`，SpecPV 评测仓 `/home/tanzhendong/SpecPV`。

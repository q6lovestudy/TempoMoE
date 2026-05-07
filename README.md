# TempoMoE

多模态 MoE 模型的快慢思考专家分析 pilot 项目。

## 目标

在 `Kimi-VL-A3B-Thinking-2506` 上，对比同一样本在不同 prompt 模式下的 MoE
路由行为：
- 3 个 fast prompt（要求直接回答）
- 3 个 slow prompt（要求 step-by-step）
- 1 个 default prompt（中性基线）

完整保存模型输出 + 每层每 token 的 top-k 路由数据，离线分析时再计算
fast/slow 专家差异。

## 目录结构

```
TempoMoE/
├── configs/
│   ├── prompts.yaml         # fast/slow/default prompt 模板
│   └── pilot.yaml           # 路径、GPU、采样、推理等配置
├── src/
│   ├── model_loader.py      # 加载 Kimi-VL，限定可见 GPU
│   ├── moe_hook.py          # MoEGate forward hook，捕获 top-k 路由
│   ├── prompts.py           # prompt 模板加载 + chat messages 构造
│   ├── data_loader.py       # RealWorldQA / MathVista 下载与索引
│   ├── inference.py         # 单样本推理 + token 段定位
│   ├── parser.py            # 解析 ◁think▷ 标签
│   └── storage.py           # 写 main_records.jsonl + npz
├── scripts/
│   ├── 00_smoke_test.py     # 单样本验证 pipeline
│   ├── 01_prepare_pilot.py  # 构建数据集索引
│   └── 02_run_pilot.py      # 跑全部样本 × 全部 prompt 变体
├── data/                    # 数据集图像 + pilot_index.jsonl
├── model/                   # 模型文件（已存在）
├── runs/                    # 实验输出
│   ├── smoke/               # smoke test 输出
│   └── pilot/
│       ├── main_records.jsonl
│       └── routes/{sample_id}__{variant_id}.npz
├── requirements.txt
└── README.md
```

## 关键设计决策（你点头过的几条）

1. **排除 shared experts**：Kimi-VL 每层有 2 个 always-on 共享专家，
   它们不走 router gate，hook 自然捕获不到 → 已自动排除。
2. **prompt-phase / decode-phase 分开记录**：phase_per_token 字段标记。
3. **modality 标记**：image / question / instruction / decode 四类，
   离线分析时可以只在 image+question token 上对比 fast/slow，避免被
   "instruction 文本本身"污染。
4. **稀疏存储 top-k**：保存 `expert_ids[L,T,K]` + `gate_weights[L,T,K]`，
   而非聚合统计。后期想换分析口径不用重跑。
5. **Greedy decoding**：do_sample=False，可复现，噪音降一个数量级。
6. **断点续跑**：`02_run_pilot.py` 跑挂了直接重跑会自动跳过已完成。

## 跑通流程

### Step 0. Smoke test（必须先过）

```bash
cd /data/zhoukeru/q/TempoMoE
python scripts/00_smoke_test.py
```

成功标志（控制台会打印）：
- `[OK] 注册了 26 个 MoEGate hook`
- `[OK] 路由数据 shape = (26, T_total, 6)`
- `[OK] 路由 phase 分布: prompt=A, decode=B`
- `[OK] modality 分布: image=I, question=Q, instruction=N, decode=D`
- 模型输出片段
- `[Smoke Test 全部通过]`

如果任一步失败，**先解决再往下走**。常见问题：
- 找不到 26 个 hook：检查 `gate_class_name` 是否为 `MoEGate`（默认就是）
- 显存不够：调小 `max_memory_per_gpu` 或换更空的卡
- modality 全 0：图像 token 没找到，可能 chat template 输出格式与预期不符

### Step 1. 准备数据

```bash
python scripts/01_prepare_pilot.py
```

第一次跑会下载 RealWorldQA 和 MathVista 的 testmini，存到
`data/pilot/images/`，索引写到 `data/pilot/pilot_index.jsonl`。

调整规模：
```bash
python scripts/01_prepare_pilot.py --rwqa_limit 50 --mvista_limit 50
```

### Step 2. 跑 pilot

```bash
python scripts/02_run_pilot.py
```

100 条样本 × 7 prompt = 700 次推理。greedy + max_new_tokens=1024，
单条预计 ~10s（slow prompt 长一些），总时长约 2 小时。

中途挂了直接重跑即可（自动跳过已完成的）。

只跑一个数据集 / 前 N 条：
```bash
python scripts/02_run_pilot.py --datasets RealWorldQA --max_samples 10
```

清空重跑：
```bash
python scripts/02_run_pilot.py --fresh
```

## 输出文件格式

### `runs/pilot/main_records.jsonl`（每行一条）

```jsonc
{
  "sample_id":            "rwqa_0042",
  "dataset":              "RealWorldQA",
  "prompt_type":          "slow",
  "variant_id":           "slow_v2",
  "question":             "...",
  "image_path":           "...",
  "ground_truth":         "...",
  "prompt_text":          "Reason through ...",
  "model_output":         "◁think▷ ... ◁/think▷ The answer is X.",
  "parsed_answer":        "X",
  "parse_status":         "ok",
  "has_think_marker":     true,
  "correct":              null,        // 离线再算
  "input_token_count":    340,
  "output_token_count":   240,
  "token_segments": {
    "image":       [0, 256],
    "question":    [312, 340],
    "instruction": [256, 312],
    "decode":      [340, 580]
  },
  "decode_config":        {...},
  "total_latency":        5.21,
  "route_file":           "rwqa_0042__slow_v2.npz"
}
```

### `runs/pilot/routes/{sample_id}__{variant_id}.npz`

```python
import numpy as np
data = np.load("runs/pilot/routes/rwqa_0042__slow_v2.npz")
data["expert_ids"]          # int16   [26, T_total, 6]
data["gate_weights"]        # float16 [26, T_total, 6]
data["phase_per_token"]     # int8    [T_total]   0=prompt 1=decode
data["modality_per_token"]  # int8    [T_total]   0=image 1=question 2=instruction 3=decode
data["chunk_lengths"]       # int32   [n_forward_calls]
```

## 离线分析的入门姿势

跑完 pilot 后，加载所有 npz 就能算 `Δr`、`Δg`、bootstrap 稳定性等：

```python
import numpy as np, json
from collections import defaultdict

# 累计每个 (prompt_type, layer, expert) 的 token 数和被选次数
counts = defaultdict(lambda: {"selected": 0, "tokens": 0, "weight_sum": 0.0})

for line in open("runs/pilot/main_records.jsonl"):
    rec = json.loads(line)
    if rec["dataset"] != "RealWorldQA":   # 先按数据集分别算
        continue
    routes = np.load(f"runs/pilot/routes/{rec['route_file']}")
    eid, gw, phase, mod = (routes["expert_ids"], routes["gate_weights"],
                            routes["phase_per_token"], routes["modality_per_token"])
    # 只看 prompt-phase + 非 instruction token（image + question）
    mask = (phase == 0) & ((mod == 0) | (mod == 1))
    L, T, K = eid.shape
    for l in range(L):
        for k in range(K):
            ids_l = eid[l, mask, k]
            wts_l = gw[l, mask, k]
            for e_id, w in zip(ids_l, wts_l):
                key = (rec["prompt_type"], l, int(e_id))
                counts[key]["selected"] += 1
                counts[key]["weight_sum"] += float(w)
            for _ in range(int(mask.sum())):
                key2 = (rec["prompt_type"], l, "_total_")
                counts[key2]["tokens"] += 1
# 然后算 Δr = r_slow - r_fast 等等
```

具体的稳定性 bootstrap、随机专家 baseline、mask/boost 因果验证留到后续阶段。

## 已知 TODO（Stage 2/3 的接口已经预留）

- [ ] 加 bootstrap 稳定性选择（SAFEx 风格）
- [ ] 加 random expert null baseline
- [ ] 加 mask/boost 因果验证：在 router 输出上做 logit suppression
- [ ] correct 字段离线计算（数学题用规则匹配 + LLM judge 双轨）
- [ ] thinking token 范围的细致 ground-truth 校验（处理标签嵌套等异常）

## 调试小技巧

- 看一条样本的路由：
  ```python
  from src.storage import load_routes
  r = load_routes("runs/pilot/routes", "rwqa_0042__slow_v2.npz")
  print(r["expert_ids"].shape)
  ```
- 检查 hook 是否抓到所有 26 层：smoke test 输出第二行
- 检查 modality 标定是否正确：smoke test 输出 modality 分布；image 段长度
  应该 = 图像 patch 数（Kimi-VL 取决于图像分辨率，通常 128~512 之间）

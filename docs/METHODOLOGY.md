# 多模态 MoE 中快慢思考专家识别 —— 方法学整理

> 项目代号：TempoMoE
> 文档目的：研究思路 + Stage 1 实验设计的完整记录，便于自己回看和向导师汇报
> 最后更新：2026-05-06

---

## 1. 研究问题

多模态 thinking 类大模型（如 Kimi-VL-A3B-Thinking）在面对**简单视觉问答题**时，会不必要地进入"慢思考"模式：生成长 CoT、有时甚至引入幻觉、绕弯子算错。同时面对**真正需要推理的复杂题**时，慢思考又是必要的。

核心问题不是"thinking 不好"，而是：

> **什么时候该快、什么时候该慢？模型内部是否已经存在"快思考专家"和"慢思考专家"的分化？**

如果答案是肯定的，下一步就能在推理时**有选择地激活/抑制**它们，把"是否进入慢思考"的决策权从生成层面下沉到 MoE 路由层面。

---

## 2. 核心想法（Stage 1 - 3 路线图）

| 阶段 | 目标 | 是否需要训练 | 状态 |
|---|---|---|---|
| **Stage 1** | 用 prompt 对比识别候选快/慢专家 | 不需要训练 | **进行中** |
| **Stage 2** | 用 mask / boost 因果验证这些专家 | 不需要训练 | 待办 |
| **Stage 3** | 在线难度感知地调节这些专家权重 | 训练轻量调节器 | 远期 |

> **本文档只详细描述 Stage 1**。Stage 2/3 在第 8、9 节给出方向，避免现在过度展开。

Stage 1 的一句话定位：

> 在不重训模型的前提下，**用三种 prompt 引导模型进入不同思考模式，对比 MoE 路由差异，识别出与"快/慢思考"行为相关的专家**。

---

## 3. 为什么不重训 router？

`Kimi-VL-A3B-Thinking-2506` 共 16B 参数，重训 router/MoE 的成本和数据需求远超个人项目能承受。
而 SteerMoE、ESFT 等 2024-2025 工作已经证明：**MoE 模型在预训练后就自然形成了功能专门化**，可以通过观测找出"行为相关专家"，再做轻量干预。

我们的工作沿着这条路走，**只是把"行为类别"从 safety/refusal 换成 fast/slow thinking，把模态从 LLM 换成 MLLM**。

---

## 4. Stage 1 详细方法

### 4.1 模型与运行环境

- **模型**：Kimi-VL-A3B-Thinking-2506（DeepSeek-V3 风格 MoE 架构）
- **关键架构参数**：
  - 27 个 transformer 层，第 0 层 dense，**第 1-26 层为 MoE（共 26 层）**
  - 每层路由专家数 N = 64，**共享专家数 = 2（always-on）**
  - top-k = 6，scoring = sigmoid，topk_method = noaux_tc
  - 路由打分被归一化并乘以 routed_scaling_factor=2.446
- **硬件**：8×L40-class GPU（46 GiB），bf16 推理，模型分布到 cuda:0/1
- **推理设置**：greedy decoding（do_sample=False），max_new_tokens=1024

### 4.2 数据集

| 数据集 | 角色 | pilot 数量 |
|---|---|---|
| **RealWorldQA** | 简单视觉问答（fast 友好） | 50 |
| **MathVista (testmini)** | 视觉数学推理（slow 友好） | 50 |

两个数据集在难度光谱的两端，可以同时用作：
- "通用快/慢专家" = 在两个数据集都稳定出现的候选
- "任务特异专家" = 只在一个数据集出现的候选

避免把"数学相关专家"误标为"慢思考专家"。

### 4.3 Prompt 设计（7 个变体）

每条样本跑 **3 fast + 3 slow + 1 default**，共 7 次推理。

| 类型 | 变体数 | 例子 |
|---|---|---|
| fast | 3 | "Answer the question directly... Do not provide step-by-step reasoning." |
| slow | 3 | "Carefully analyze the image and the question step by step before answering." |
| default | 1 | "Answer the question based on the image." |

**为什么用 3 个变体而不是 1 个**：
- 避免特定 prompt 写法的 idiosyncratic 效应（某个词刚好特别激活某个专家）
- 后续稳定性检查的 baseline：3 个 fast 变体下方向都一致才算稳定信号
- default 提供"中性基线"，未来分析模型自然倾向时用

完整 prompt 见 `configs/prompts.yaml`。

### 4.4 推理与路由记录（hook 设计）

**Hook 位置**：每个 MoE 层的 `MoEGate` 模块。

`MoEGate.forward(hidden_states)` 返回 `(topk_idx, topk_weight, aux_loss)`，hook 在每次 forward 调用时拦截前两个张量：

- `topk_idx`：[T, K=6] 选中的专家 id
- `topk_weight`：[T, K=6] 归一化后的门权重

**Phase 区分**：
- 第一次 forward = prefill，T = 输入 token 数
- 后续每次 forward = decode 一步，T = 1
- 用 chunk 长度自动标记 phase（详见 `src/moe_hook.py`）

**Shared expert 自动排除**：共享专家不走 gate，hook 自然抓不到 → 不会污染统计。

### 4.5 Token 段切分（4 段）

这是整个方法学最关键的设计之一。Chat template 渲染后的 prompt 结构：

```
[系统前缀] [<|im_user|>] [image tokens × ~1768] [instruction text] [question text] [<|im_end|>] ...
                              ↑                    ↑
                         fast/slow 输入完全相同     开始有差异
```

我们把每个 token 标记为以下 4 种 modality：

| modality | 在 fast vs slow 下的差异 | 信号干净度 |
|---|---|---|
| **image** | 输入 + KV cache 完全相同 → routing 完全相同 | **0**（噪声底校准） |
| **question** | 输入文本相同，仅 KV cache 有 instruction 影响 | **小但干净** ★ |
| **instruction** | 字面文本就不同 | 大但被字面 token 污染 |
| **decode** | 完全不同的生成内容 | 最大但被长度+内容污染 |

**为什么要分段**：
- 不同段的"信号干净度"截然不同；混在一起算 Δr 会得到误导性结论
- IMAGE 段是天然的 **null baseline**：理论上 Δr = 0，实测 ~ 0.001（bf16 跨卡浮点噪声），用来校准噪声底
- QUESTION 段是**最干净的真实信号**：输入相同，差异完全来自 fast/slow 在 KV cache 上的传递
- 后续 03 脚本对每段分别算 Δr，对每段分别筛候选

边界定位用特殊 token 锚定（`<|media_pad|>`、`<|im_end|>`），避开 BPE 上下文敏感性导致的子序列匹配失败。

### 4.6 核心指标

对每个 (dataset, prompt_type, segment, layer, expert) 累加：

```
count_per_expert[e]      = Σ_t  1[e ∈ TopK(t,l)]
weight_sum_per_expert[e] = Σ_t  g_{e,t,l} · 1[e ∈ TopK(t,l)]
total_tokens             = Σ_t  1
```

然后两个互补的核心指标：

```
selection_rate r_{l,e}    = count_e / total_tokens         (ESFT-Token 风格)
mean_gate_weight g_{l,e}  = weight_sum_e / count_e         (ESFT-Gate 风格)
```

**Δr = r_slow - r_fast**（SteerMoE 风格的 risk-difference）
**Δg = g_slow - g_fast**

要求两个指标**方向一致**才纳入候选，是免费的稳健性检查。

### 4.7 Bootstrap 稳定性（SAFEx 风格）

只算一次 Δr 不够稳健 —— 换一批样本、换一个 prompt 变体，排名可能完全翻车（NeurIPS 2025 SAFEx 已论证）。

**Bootstrap 流程**：
1. 跑 B = 50 轮
2. 每轮随机取 50% 样本（按 sample_id 而非按 record，保留所有变体共变）
3. 每轮重算每段每层的 Δr，记录 |Δr| 排前 K=10 的专家
4. 累计每个专家在多少轮入选 top-K → 入选率 π = 计数 / B
5. **π ≥ 0.7** 的视为稳定候选

由于 04 脚本复用 03 阶段已构建的 per_sample 累加器，bootstrap 不需要重新读 700 个 npz，速度可控。

### 4.8 跨数据集一致性筛选

用 RealWorldQA 和 MathVista 分别跑稳定性，再取交集：

| 类别 | 含义 | 论文价值 |
|---|---|---|
| **通用候选**（universal） | 在两个数据集都稳定，且方向一致 | **最强证据**，论文重点 |
| RealWorldQA-only | 只在 RWQA 稳定 | 任务特异性提示 |
| MathVista-only | 只在 MathVista 稳定 | 同上 |

避免"在 MathVista 上发现的慢专家其实是数学相关专家"这类陷阱。

### 4.9 候选筛选规则（最终交付）

`05_select_candidates.py` 在 03 + 04 的基础上加最后一道筛：

- 只看 question + decode 段（image 是噪声底无意义；instruction 含字面干扰）
- π ≥ 0.7（默认；可调）
- |mean Δr| ≥ 0.005（默认；可调）

输出：

| 文件 | 说明 |
|---|---|
| `candidates.csv` | 全部通过筛选的候选 |
| **`candidates_universal.csv`** | **跨数据集一致的通用候选 ← Stage 1 最终交付物** |

---

## 5. 实验流程总览图

```
[100 样本 × 7 prompt = 700 次推理]
         │
         │  scripts/02_run_pilot.py
         ▼
┌─────────────────────────────────┐
│ runs/pilot/main_records.jsonl   │  每行：sample_id, prompt_type, model_output, token_segments...
│ runs/pilot/routes/*.npz         │  每文件：[26层 × T_token × top6] expert_id + gate_weight
└──────────────┬──────────────────┘
               │
               │  scripts/03_compute_expert_scores.py
               ▼
┌─────────────────────────────────┐
│ analysis/expert_scores*.csv     │  每 (dataset, segment, layer, expert) 一行 r/g/Δr/Δg
└──────────────┬──────────────────┘
               │
               │  scripts/04_bootstrap_stability.py (B=50)
               ▼
┌─────────────────────────────────┐
│ analysis/stability_scores.csv   │  每 (dataset, segment, layer, expert) 的 π
└──────────────┬──────────────────┘
               │
               │  scripts/05_select_candidates.py
               ▼
┌─────────────────────────────────┐
│ analysis/candidates.csv         │  通过 π+Δr 双重筛选
│ analysis/candidates_universal   │  ★ 跨数据集一致候选
└──────────────┬──────────────────┘
               │
               │  scripts/06_visualize.py
               ▼
   runs/pilot/figures/  fig1-fig5
```

---

## 6. 关键设计决策（Q&A 风格 / 老师可能问的）

#### Q1：你的工作和 SteerMoE / ESFT 区别在哪？
- SteerMoE 在纯 LLM 上找 safety/refusal 专家
- ESFT 在 LLM 上找任务专门化专家
- **本文工作首次把"行为相关专家识别 + 因果干预"应用到多模态 thinking 模型，目标是 fast/slow 思考模式**
- 方法学借鉴它们（risk-difference + bootstrap stability + 双向干预），但研究问题不同

#### Q2：你怎么定义"快/慢专家"？
- 操作定义：在 fast prompt 下 token 选中率 r 显著低于 slow prompt 下的 r 的专家 = 慢专家候选
- 形式上：Δr_{l,e} = r_slow_{l,e} - r_fast_{l,e}，Δr > 阈值 + 通过 bootstrap 稳定性 = 慢候选
- 不是绝对定义，而是 **fast/slow 引导下的相对偏好**

#### Q3：怎么验证找到的就是真的快/慢专家不是噪声？
四层防御：
1. **IMAGE 段噪声底校准**：理论 Δr=0，实测 ~0.001，作为"信号必须高于此值"的基线
2. **Bootstrap 稳定性 (π ≥ 0.7)**：50 轮里至少 35 轮入选 top-K
3. **跨数据集一致**：RealWorldQA 和 MathVista 都通过筛选
4. **Stage 2 因果验证**：mask/boost 后 accuracy 和 output 长度按预期变化（待做）

#### Q4：为什么 3 个 prompt 变体？1 个不够吗？
单一 prompt 写法可能因为某个词（如 "directly"）特别激活某个专家，造成假阳性。3 个语义相同但表达不同的变体下都通过，才能区分"prompt 表面词触发"和"思考模式触发"。

#### Q5：Δr 算多大才有意义？
- IMAGE 段噪声底约 0.001（bf16 + 跨卡的浮点噪声，已实测）
- QUESTION 段实测 0.005-0.013（**5-10 倍噪声底**，干净信号）
- DECODE 段实测 0.018-0.042（**15-40 倍噪声底**，强信号）
- INSTRUCTION 段实测 0.022-0.049（**20-50 倍噪声底**）

绝对值看似小，但相对噪声底显著。最终筛选用 |Δr| ≥ 0.005 = ≥5×噪声底。

#### Q6：为什么要把 token 切成 4 段？整体算不行吗？
不行。如果 image+question+instruction+decode 一起算：
- image 段 1768 个 token 全是零信号，会把真信号稀释
- decode 长度差异巨大（fast ~50 / slow ~120），混合算会被长度比例主导
- instruction 段字面文本不同，会把信号高估为"思考模式差异"

分段后每段独立分析，能干净地看出信号在哪、有多大、是否被某种 confound 解释。

#### Q7：为什么排除 shared expert？
共享专家在 Kimi-VL 的 MoE 层是 always-on 的（不走 gate 竞争）。它们对所有 token 都激活，无法做 fast/slow 对比。Hook 只挂在 `MoEGate` 上，自然不会捕获 shared expert，符合方法学需要。

---

## 7. 已规避的方法学陷阱

| # | 陷阱 | 我们的处理 |
|---|---|---|
| 1 | shared expert 污染 | hook 自然排除 |
| 2 | prompt-phase 和 decode-phase 混合 | 分别记录 phase tag |
| 3 | 长度 confound（slow 输出长 → naive sum 偏） | 全用 token 数归一化的 r 和 g |
| 4 | 单一 prompt 写法 idiosyncrasy | 3 个语义等价变体 |
| 5 | 任务特异专家被误认为通用 | RealWorldQA × MathVista 跨数据集交集 |
| 6 | 子序列匹配在 BPE 上下文敏感时失败 | 用特殊 token 锚定段边界 |
| 7 | 随机解码引入伪信号 | 全程 greedy decoding |
| 8 | 单次跑出来的排名不稳 | SAFEx bootstrap 稳定性选择 |
| 9 | 把"指令文本字面差异"误认为"思考模式差异" | 主分析只在 question + decode 段 |
| 10 | 浮点 / 多卡非确定性看起来像信号 | image 段做 null check 校准噪声底 |

---

## 8. Stage 2 预告（因果验证）

拿 `candidates_universal.csv` 里的候选专家，做双向 logit 干预：

**Mask（抑制慢专家）**：
- 在 router 输出的 log-softmax 上，对候选慢专家减去 δ ∈ {1, 2, 4, 8}
- 在 RealWorldQA 上测：accuracy 是否保持？output 长度是否下降？think token 数是否减少？
- **预期**：accuracy 不降甚至略升（减少 overthinking），长度明显下降

**Boost（强化慢专家）**：
- 同样位置加 δ
- 在 fast prompt 下测：原本短的输出会不会变长？会不会出现 CoT？
- **预期**：fast prompt 下也产生明显 think 段

**Random expert null baseline**：
- 同样大小的随机路由专家集合做对照
- 我们的候选必须显著优于 random，才算通过因果验证

**Hard 题验证**：
- 同样的 mask/boost 在 MathVista 上做
- mask 慢专家应该让 MathVista accuracy **下降**（说明慢专家对复杂推理是必要的）
- 这个反向证据比"简单题上 mask 不掉点"更强

---

## 9. Stage 3 远景（在线难度感知调节）

如果 Stage 2 跑通，最终目标是把"是否进入慢思考"自动化：

```
                 [图像 + 问题]
                     │
                     ▼
          ┌──────────────────┐
          │ 难度感知调节器     │  ← 轻量 MLP，输入图像/问题特征
          └────────┬─────────┘
                   │ 输出 β_l (每层调节强度)
                   ▼
           原 router logit z_{l,e}
                   ↓
           z'_{l,e} = z_{l,e} + β_l × s_{l,e}
                   ↓                   （s_{l,e} = Stage 1 找到的快慢属性）
            top-K + softmax
```

简单题 → β_l 偏负 → 强化快专家 → 直接给答案
难题 → β_l 偏正 → 强化慢专家 → 进入 CoT

这一步需要训练 β_l，但只训这一个轻量调节器，不动模型本体。

---

## 10. 当前进度（2026-05-06）

- ✓ 项目代码框架搭建完成（src/ + scripts/）
- ✓ Smoke test 通过（模型加载、hook、生成、保存全链路 OK）
- ✓ 3 样本 sanity check：IMAGE 段噪声底 ~0.001、QUESTION/DECODE 信号 0.01-0.04，符合方法学预期
- ✓ 离线分析脚本（03-06）已实现并通过语法检查
- **进行中**：100 样本 × 7 prompt = 700 次推理的全量 pilot（约 2 小时）
- 待办：
  - [ ] pilot 跑完后跑 03-06 离线分析
  - [ ] 检查 universal candidates 数量与质量
  - [ ] 写 Stage 1 阶段性总结
  - [ ] 进入 Stage 2 因果验证

---

## 11. 相关工作定位

| 工作 | 它做什么 | 我们的差异 |
|---|---|---|
| **SteerMoE** (2025) | LLM 上 safety/refusal 专家识别 + steering | 行为类别（fast/slow 而非 safety），模态（MLLM 而非 LLM） |
| **ESFT** (DeepSeek, 2024) | 基于 gate score 找任务专门化专家用于 LoRA | 找思考模式而非任务，不做 LoRA 而做干预 |
| **SAFEx** (NeurIPS 2025) | LLM safety expert 的 bootstrap 稳定选择 | 借鉴方法（bootstrap stability），换问题 |
| **Routing Distraction** (2026) | MLLM 上图像 vs 文本的路由分布对比 | 最相关；他们对比模态，我们对比思考模式 |
| **Metis-HOME** | MLLM hybrid thinking branch routing | 在 query/branch 层面控制，我们在专家层面 |
| **DAREe**, **Harder Tasks Need More Experts** | 难度感知 MoE 路由 | 我们在 Stage 3 才会接近这条线 |

---

## 12. 一句话总结

> **通过 fast/slow prompt 在不同 token 段引发的路由差异，用 SteerMoE 风格的 Δr 打分 + SAFEx 风格的 bootstrap 稳定性 + 跨数据集一致性三层筛选，识别多模态 MoE thinking 模型中的快慢思考专家，作为后续因果干预（mask/boost）和在线难度感知调节的基础。**

---

## 附录 A：术语表

| 术语 | 定义 |
|---|---|
| MoE | Mixture-of-Experts |
| router / gate | 决定每个 token 走哪几个 expert 的小网络 |
| top-k | 每 token 选 k 个 expert 参与计算（Kimi-VL: k=6） |
| shared expert | always-on 的 expert，不走 router 竞争 |
| routed expert | 走 router 竞争的 expert（Kimi-VL: 64 个） |
| prompt-phase / prefill | 输入序列一次性 forward 的阶段 |
| decode-phase | 一次生成一个 token 的阶段 |
| Δr (delta_r) | r_slow - r_fast，专家选中率的 fast/slow 差 |
| Δg (delta_g) | g_slow - g_fast，平均门权的 fast/slow 差 |
| π (stability) | bootstrap 中专家入选 top-K 的轮次比例 |
| candidate | 通过筛选的 fast/slow 候选专家 |
| universal candidate | 跨数据集一致的候选 |

## 附录 B：项目目录索引

```
TempoMoE/
├── configs/
│   ├── pilot.yaml          实验配置
│   └── prompts.yaml        7 个 prompt 变体
├── src/
│   ├── model_loader.py     加载 Kimi-VL
│   ├── moe_hook.py         MoEGate forward hook
│   ├── prompts.py          prompt 构造
│   ├── data_loader.py      数据集加载
│   ├── inference.py        单样本推理 + token 段定位
│   ├── parser.py           解析 ◁think▷ 标签
│   ├── storage.py          写 jsonl + npz
│   └── analysis_utils.py   离线分析公共工具
├── scripts/
│   ├── 00_smoke_test.py        Pipeline 单样本验证
│   ├── 01_prepare_pilot.py     下载 RWQA + MathVista
│   ├── 02_run_pilot.py         全量推理
│   ├── 03_compute_expert_scores.py    算 Δr / Δg
│   ├── 04_bootstrap_stability.py      bootstrap 稳定性
│   ├── 05_select_candidates.py        最终候选筛选
│   └── 06_visualize.py                出图
└── docs/
    └── METHODOLOGY.md      本文档
```

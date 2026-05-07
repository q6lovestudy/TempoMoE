"""
MoE 路由干预模块（Stage 2 因果验证用）

核心思想：
    在 MoEGate 上注册 forward hook，劫持 (topk_idx, topk_weight, aux_loss) 输出，
    对"目标专家"的门权进行修改（suppress / boost），返回新的 (topk_idx, topk_weight)。

两种干预：
    1. SUPPRESS（mask）: 把目标专家的 topk_weight 乘以 (1 - factor)
       factor=1.0 → 完全 mask（权重置 0）
       factor=0.5 → 软抑制（权重减半）
       这种方式只影响"已被 router 选中的"目标专家。

    2. BOOST: 同样修改 topk_weight，乘以 (1 + factor)；可让目标专家在被选中时贡献更大。
       严格意义上的 boost（让原本没选中的专家被选上）需要 monkey-patch
       MoEGate.forward 重做 top-K，比较复杂；此处 v1 只实现 suppress 这种最关键的实验。

使用：
    suppressor = ExpertSuppressor(model, layer_to_experts={8: {6}, 22: {32}}, factor=1.0)
    suppressor.attach()
    out = model.generate(...)        # 这次推理里，layer 8 的 e6 / layer 22 的 e32 被 mask
    suppressor.detach()
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 主干预类
# ----------------------------------------------------------------------
class ExpertSuppressor:
    """
    在每个 MoEGate 上注册 hook，对 topk_weight 做 in-place 修改。

    factor 含义：
        suppress 模式 → 目标专家 weight *= (1 - factor)
        boost    模式 → 目标专家 weight *= (1 + factor)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_to_experts: Dict[int, Set[int]],
        factor: float = 1.0,
        mode: str = "suppress",
        gate_class_name: str = "MoEGate",
    ):
        """
        Args:
            model:             加载好的 Kimi-VL
            layer_to_experts:  {moe_layer_idx (0..25) → set of expert ids to intervene}
            factor:            干预强度。suppress: 0~1。boost: 任意正数（如 0.5/1.0/2.0）
            mode:              "suppress" 或 "boost"
            gate_class_name:   门控模块的类名（默认匹配 MoEGate）
        """
        assert mode in ("suppress", "boost"), f"未知 mode={mode}"
        if mode == "suppress":
            assert 0.0 <= factor <= 1.0, f"suppress 模式 factor 应在 [0,1]，实际 {factor}"
        else:
            assert factor >= 0.0, f"boost 模式 factor 应 >= 0，实际 {factor}"

        self.model = model
        self.layer_to_experts = {int(k): set(int(e) for e in v)
                                  for k, v in layer_to_experts.items()}
        self.factor = float(factor)
        self.mode = mode
        self.gate_class_name = gate_class_name
        self._hook_handles = []
        # 记录每层是否真的有干预目标，便于打印统计
        self._n_target_per_layer: Dict[int, int] = {}

    def attach(self) -> int:
        """注册 hook。返回安装了 hook 的 MoE 层数（无目标专家的层不安装）。"""
        if self._hook_handles:
            raise RuntimeError("已有 hook 注册，请先 detach()")

        # 按发现顺序给每个 MoEGate 分配 layer_idx (0..25)
        gate_modules = []
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == self.gate_class_name:
                gate_modules.append((name, module))

        if not gate_modules:
            raise RuntimeError(f"找不到类名 {self.gate_class_name} 的模块")

        n_installed = 0
        n_total_targets = 0
        for layer_idx, (name, module) in enumerate(gate_modules):
            target_experts = self.layer_to_experts.get(layer_idx, set())
            if not target_experts:
                continue   # 该层无目标，不装 hook
            handle = module.register_forward_hook(
                self._make_hook(layer_idx, target_experts)
            )
            self._hook_handles.append(handle)
            self._n_target_per_layer[layer_idx] = len(target_experts)
            n_installed += 1
            n_total_targets += len(target_experts)

        logger.info(
            f"[Suppressor] mode={self.mode} factor={self.factor} | "
            f"安装在 {n_installed} 层 / 共 {len(gate_modules)} 层 | "
            f"涉及目标专家总数 {n_total_targets}"
        )
        return n_installed

    def detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._n_target_per_layer.clear()

    # ------------------------------------------------------------------
    def _make_hook(self, layer_idx: int, target_experts: Set[int]):
        """为某层生成 hook 闭包。"""
        target_list = sorted(target_experts)
        # 用 buffer 形式存到 device 上，避免每次 forward 都重新搬
        target_tensor = torch.tensor(target_list, dtype=torch.long)

        if self.mode == "suppress":
            multiplier = 1.0 - self.factor   # factor=1 → multiplier=0（hard mask）
        else:  # boost
            multiplier = 1.0 + self.factor

        def hook(module, inputs, output):
            # MoEGate.forward 返回 (topk_idx, topk_weight, aux_loss)
            topk_idx, topk_weight, aux_loss = output
            # topk_idx: [bsz*seq_len, K]
            # topk_weight: [bsz*seq_len, K]

            tt = target_tensor.to(topk_idx.device)
            # mask: True where this slot's expert is in target set
            in_target = torch.isin(topk_idx, tt)   # [bsz*seq_len, K] bool

            if multiplier == 0.0:
                # hard mask 情况：直接乘 0
                new_weight = topk_weight.masked_fill(in_target, 0.0)
            else:
                new_weight = torch.where(
                    in_target,
                    topk_weight * multiplier,
                    topk_weight,
                )

            return topk_idx, new_weight, aux_loss

        return hook


# ----------------------------------------------------------------------
# 候选专家集合的构造工具
# ----------------------------------------------------------------------
def load_candidates_to_layer_dict(
    csv_path: str,
    leaning: str = None,
    segments: Iterable[str] = None,
    top_n_by_score: int = None,
    score_fn=None,
) -> Dict[int, Set[int]]:
    """
    从 candidates_universal.csv 读出一个 {layer_idx → set(expert_ids)} 字典。

    Args:
        csv_path:        candidates_universal.csv 路径
        leaning:         "slow" / "fast" / None（不过滤）
        segments:        如 ["question", "decode"]，None 表示全部
        top_n_by_score:  只保留 top-N（按 score_fn 排序）
        score_fn:        给每行打分的函数，默认 = π × |Δr|

    Returns:
        {layer_idx (0..25): {expert_id, ...}}
        注意：同一 (layer, expert) 在 question 和 decode 段都出现时只保留一次。
    """
    import csv as _csv

    if score_fn is None:
        def score_fn(r):
            return float(r["stability_pi_avg"]) * abs(float(r["mean_delta_r_avg_across_datasets"]))

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if leaning and r["leaning"] != leaning:
                continue
            if segments and r["segment"] not in segments:
                continue
            r["_score"] = score_fn(r)
            rows.append(r)

    rows.sort(key=lambda x: -x["_score"])

    # 去重：同一 (layer, expert) 只保留分数最高的那行
    seen: Set[Tuple[int, int]] = set()
    deduped = []
    for r in rows:
        key = (int(r["layer"]), int(r["expert"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    if top_n_by_score is not None:
        deduped = deduped[:top_n_by_score]

    layer_to_experts: Dict[int, Set[int]] = defaultdict(set)
    for r in deduped:
        layer_to_experts[int(r["layer"])].add(int(r["expert"]))

    return dict(layer_to_experts)


def make_random_baseline(
    layer_to_experts: Dict[int, Set[int]],
    n_routed_experts: int = 64,
    n_layers: int = 26,
    seed: int = 42,
    exclude_layer_to_experts: Dict[int, Set[int]] = None,
) -> Dict[int, Set[int]]:
    """
    给一个目标集合（如真候选），构造一个同样规模的随机对照集合。
    每层随机抽取相同数量的非目标专家。

    Args:
        layer_to_experts:        要匹配规模的真目标集合
        n_routed_experts:        每层路由专家总数（Kimi-VL: 64）
        n_layers:                MoE 层数（Kimi-VL: 26）
        seed:                    随机种子（保证可复现）
        exclude_layer_to_experts:  必须排除的专家集合（避免与真候选有重叠）

    Returns:
        与 layer_to_experts 同规模的随机集合
    """
    import random

    rng = random.Random(seed)
    exclude_layer_to_experts = exclude_layer_to_experts or {}
    out: Dict[int, Set[int]] = {}
    for layer_idx, target_set in layer_to_experts.items():
        n_target = len(target_set)
        excluded = exclude_layer_to_experts.get(layer_idx, set()) | target_set
        candidates = [e for e in range(n_routed_experts) if e not in excluded]
        if n_target > len(candidates):
            raise RuntimeError(
                f"layer {layer_idx} 候选不够：要 {n_target} 个，可选 {len(candidates)} 个"
            )
        out[layer_idx] = set(rng.sample(candidates, n_target))
    return out


def summarize_layer_dict(d: Dict[int, Set[int]]) -> str:
    """打印用：把 layer_to_experts 摘要成一行文字。"""
    n_layers = len(d)
    n_total = sum(len(v) for v in d.values())
    layer_list = sorted(d.keys())
    return (
        f"涉及 {n_layers} 层，共 {n_total} 个专家。"
        f" 层分布: " +
        ",".join(f"L{l}:{len(d[l])}" for l in layer_list)
    )

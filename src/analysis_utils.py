"""
离线分析公共工具。

数据流：
  main_records.jsonl + routes/*.npz
    │
    ▼
  ExpertAccumulator: 按 (dataset, prompt_type, segment, layer[, sample_id]) 聚合
    │
    ▼
  selection_rate / mean_gate_weight → Δr / Δg → 候选专家

术语：
  segment       : "image" / "question" / "instruction" / "decode"
  prompt_type   : "fast" / "slow" / "default"
  layer         : 0..25（这里指 MoE 层的索引，对应模型里 layer 1..26 的 MoE）
  expert        : 0..63（routed expert id；shared expert 已被 hook 自动排除）

注意：
  - selection_rate r_e = (有多少 token 把 e 选进 top-K) / (总 token 数)
    所以 sum_e r_e = K = 6，每个 r_e ∈ [0, 1]
  - mean_gate_weight g_e = e 被选中时的平均门权
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---- 模型常量（Kimi-VL-A3B-Thinking-2506） ----
N_LAYERS = 26          # 路由层数（first_k_dense_replace=1，所以 26 层 MoE）
N_EXPERTS = 64         # 每层路由专家数
TOP_K = 6              # 每 token 选择的专家数

# segment 名 ↔ modality 编号
SEGMENT_NAMES = ["image", "question", "instruction", "decode"]
SEGMENT_CODES = {"image": 0, "question": 1, "instruction": 2, "decode": 3, "other": 4}


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def load_main_records(path: str) -> List[dict]:
    """读 main_records.jsonl 全量。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_routes(record: dict, routes_dir: str) -> dict:
    """加载一条记录对应的 npz。返回 dict 包含 expert_ids/gate_weights/phase_per_token/modality_per_token。"""
    p = Path(routes_dir) / record["route_file"]
    data = np.load(p, allow_pickle=False)
    return {k: data[k] for k in data.files}


def make_segment_mask(routes: dict, segment: str) -> np.ndarray:
    """生成给定 segment 的 token mask (布尔 [T_total])。

    - "image"/"question"/"instruction": prompt-phase 内对应 modality 的 token
    - "decode": 全部 decode-phase token
    """
    phase = routes["phase_per_token"]
    mod = routes["modality_per_token"]
    if segment == "decode":
        return phase == 1
    code = SEGMENT_CODES[segment]
    return (phase == 0) & (mod == code)


# ----------------------------------------------------------------------
# 累加器
# ----------------------------------------------------------------------
@dataclass
class ExpertAccumulator:
    """单个 (dataset, prompt_type, segment, layer[, sample_id]) 的统计累加。"""
    count_per_expert: np.ndarray = field(
        default_factory=lambda: np.zeros(N_EXPERTS, dtype=np.int64)
    )
    weight_sum_per_expert: np.ndarray = field(
        default_factory=lambda: np.zeros(N_EXPERTS, dtype=np.float64)
    )
    total_tokens: int = 0

    def add(self, expert_ids_layer: np.ndarray, gate_weights_layer: np.ndarray) -> None:
        """累加单层在某段 token 上的 routing 数据。

        Args:
            expert_ids_layer:    int16/int  [T_masked, K]
            gate_weights_layer:  float16/float [T_masked, K]
        """
        T_masked = expert_ids_layer.shape[0]
        if T_masked == 0:
            return
        flat_ids = expert_ids_layer.flatten().astype(np.int64)
        flat_wts = gate_weights_layer.flatten().astype(np.float64)
        self.count_per_expert += np.bincount(flat_ids, minlength=N_EXPERTS)[:N_EXPERTS]
        self.weight_sum_per_expert += np.bincount(
            flat_ids, weights=flat_wts, minlength=N_EXPERTS
        )[:N_EXPERTS]
        self.total_tokens += T_masked

    def add_other(self, other: "ExpertAccumulator") -> None:
        """合并另一个累加器（用于 bootstrap 子集合并）。"""
        self.count_per_expert += other.count_per_expert
        self.weight_sum_per_expert += other.weight_sum_per_expert
        self.total_tokens += other.total_tokens

    def selection_rate(self) -> np.ndarray:
        """每专家选中率: r_e = count_e / total_tokens。"""
        if self.total_tokens == 0:
            return np.zeros(N_EXPERTS, dtype=np.float64)
        return self.count_per_expert / self.total_tokens

    def mean_gate_weight(self) -> np.ndarray:
        """每专家选中时的平均门权: g_e = weight_sum_e / count_e。"""
        out = np.zeros(N_EXPERTS, dtype=np.float64)
        nonzero = self.count_per_expert > 0
        out[nonzero] = (
            self.weight_sum_per_expert[nonzero] / self.count_per_expert[nonzero]
        )
        return out


# ----------------------------------------------------------------------
# 主聚合函数
# ----------------------------------------------------------------------
def aggregate_records(
    records: List[dict],
    routes_dir: str,
    segments: List[str] = None,
    per_sample: bool = False,
    dataset_filter: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Dict[Tuple, ExpertAccumulator]:
    """遍历所有 records，按 key 累加。

    Args:
        records:        load_main_records() 的输出
        routes_dir:     npz 目录
        segments:       要分析哪些段，默认全部
        per_sample:     True 时 key 多一维 sample_id（bootstrap 必需）
        dataset_filter: 只保留这些数据集
        verbose:        打印进度

    Returns:
        dict; key 形如 (dataset, prompt_type, segment, layer) 或加上 sample_id
    """
    if segments is None:
        segments = SEGMENT_NAMES

    acc: Dict[Tuple, ExpertAccumulator] = defaultdict(ExpertAccumulator)
    n_processed = 0
    n_skipped = 0

    for rec in records:
        if dataset_filter is not None and rec["dataset"] not in dataset_filter:
            continue
        try:
            routes = load_routes(rec, routes_dir)
        except FileNotFoundError:
            n_skipped += 1
            continue

        eid = routes["expert_ids"]   # [L, T, K]
        gw = routes["gate_weights"]  # [L, T, K]
        L = eid.shape[0]

        for seg in segments:
            mask = make_segment_mask(routes, seg)
            if not mask.any():
                continue
            ids_seg = eid[:, mask, :]   # [L, T_masked, K]
            wts_seg = gw[:, mask, :]
            for l in range(L):
                if per_sample:
                    key = (rec["dataset"], rec["prompt_type"], seg, l, rec["sample_id"])
                else:
                    key = (rec["dataset"], rec["prompt_type"], seg, l)
                acc[key].add(ids_seg[l], wts_seg[l])

        n_processed += 1
        if verbose and n_processed % 100 == 0:
            print(f"  已处理 {n_processed} 条记录...")

    if verbose:
        print(f"[aggregate] 处理 {n_processed} 条，跳过（找不到 npz） {n_skipped} 条")
    return dict(acc)


def merge_per_sample(
    per_sample_acc: Dict[Tuple, ExpertAccumulator],
    sample_ids: Optional[Set[str]] = None,
) -> Dict[Tuple, ExpertAccumulator]:
    """把 per_sample 累加器合并为 (dataset, prompt_type, segment, layer) 级。

    用于 bootstrap：传入 sample_ids 指定本轮选中的样本，
    返回这个子集的合并结果。

    Args:
        per_sample_acc: aggregate_records(per_sample=True) 的输出
        sample_ids:     None 表示用全部样本

    Returns:
        dict[(dataset, prompt_type, segment, layer)] -> ExpertAccumulator
    """
    merged: Dict[Tuple, ExpertAccumulator] = defaultdict(ExpertAccumulator)
    for key, a in per_sample_acc.items():
        ds, pt, seg, layer, sid = key
        if sample_ids is not None and sid not in sample_ids:
            continue
        merged[(ds, pt, seg, layer)].add_other(a)
    return dict(merged)


# ----------------------------------------------------------------------
# 便利函数：从合并后的累加器算出 Δr / Δg 矩阵
# ----------------------------------------------------------------------
def compute_delta_matrix(
    merged_acc: Dict[Tuple, ExpertAccumulator],
    dataset: str,
    segment: str,
    metric: str = "delta_r",
) -> np.ndarray:
    """返回 [N_LAYERS, N_EXPERTS] 的 Δr 或 Δg 矩阵 (slow - fast)。

    Args:
        merged_acc: merge_per_sample() 输出，或 aggregate_records(per_sample=False) 输出
        dataset:    "RealWorldQA" / "MathVista"
        segment:    "image"/"question"/"instruction"/"decode"
        metric:     "delta_r" 或 "delta_g"
    """
    out = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.float64)
    for l in range(N_LAYERS):
        a_fast = merged_acc.get((dataset, "fast", segment, l))
        a_slow = merged_acc.get((dataset, "slow", segment, l))
        if a_fast is None or a_slow is None:
            continue
        if metric == "delta_r":
            out[l] = a_slow.selection_rate() - a_fast.selection_rate()
        elif metric == "delta_g":
            out[l] = a_slow.mean_gate_weight() - a_fast.mean_gate_weight()
        else:
            raise ValueError(f"未知 metric: {metric}")
    return out


def list_sample_ids(per_sample_acc: Dict[Tuple, ExpertAccumulator]) -> List[str]:
    """从 per_sample 累加器里抽出全部 sample_id。"""
    return sorted({key[4] for key in per_sample_acc.keys()})


def list_datasets(per_sample_acc: Dict[Tuple, ExpertAccumulator]) -> List[str]:
    """同上，抽出 dataset 列表。"""
    return sorted({key[0] for key in per_sample_acc.keys()})

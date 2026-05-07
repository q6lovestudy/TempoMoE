"""
计算 fast/slow/default 下每层每专家的选中率 r / 平均门权 g 及其 Δ。

输入：
  runs/pilot/main_records.jsonl
  runs/pilot/routes/*.npz

输出：
  runs/pilot/analysis/expert_scores.csv             - 全部数据集合并
  runs/pilot/analysis/expert_scores_RealWorldQA.csv
  runs/pilot/analysis/expert_scores_MathVista.csv

每行字段：
  dataset, segment, layer, expert,
  r_fast, r_slow, r_default,           # 选中率 r_e = count_e / total_tokens
  g_fast, g_slow, g_default,           # 选中时平均门权 g_e
  delta_r, delta_g,                    # slow - fast
  n_tokens_fast, n_tokens_slow, n_tokens_default,
  n_samples_fast, n_samples_slow, n_samples_default

跑法：
    python scripts/03_compute_expert_scores.py
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import (
    N_LAYERS, N_EXPERTS, SEGMENT_NAMES,
    load_main_records, aggregate_records, merge_per_sample,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main_records_path = cfg["paths"]["main_records"]
    routes_dir = cfg["paths"]["routes_dir"]
    out_dir = Path(args.out_dir or (PROJECT_ROOT / "runs" / "pilot" / "analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读 records: {main_records_path}")
    records = load_main_records(main_records_path)
    print(f"[INFO] 总 records: {len(records)}")

    datasets = sorted({r["dataset"] for r in records})
    print(f"[INFO] 数据集: {datasets}")

    # 用 per_sample=True 聚合一次，后面 04 也能复用同一份数据
    print("[INFO] 聚合中（per_sample=True，便于 04 bootstrap 复用）...")
    per_sample = aggregate_records(records, routes_dir, per_sample=True)
    print(f"[INFO] per_sample 累加器条目: {len(per_sample)}")

    # 全数据集合并
    full_merged = merge_per_sample(per_sample, sample_ids=None)
    write_scores_csv(full_merged, per_sample, out_dir / "expert_scores.csv")

    # 每个数据集单独输出
    for ds in datasets:
        # 只保留这个数据集的 sample_ids
        sample_ids = {key[4] for key in per_sample if key[0] == ds}
        ds_merged = merge_per_sample(per_sample, sample_ids=sample_ids)
        # 但合并函数不区分 dataset，只看 sample_id；因为不同 dataset 的 sample_id
        # 命名上有区别（rwqa_xxx / mvista_xxx），所以这里安全；保险起见再过滤一下：
        ds_merged = {k: v for k, v in ds_merged.items() if k[0] == ds}
        write_scores_csv(ds_merged, per_sample, out_dir / f"expert_scores_{ds}.csv")

    # Sanity 报告
    print_sanity_report(full_merged, datasets)
    print()
    print(f"[OK] 全部 CSV 已写入: {out_dir}")


# ----------------------------------------------------------------------
def write_scores_csv(merged, per_sample, path):
    """从合并后的累加器写 CSV。"""
    # 收集所有 (dataset, segment, layer)
    keys = sorted({(k[0], k[2], k[3]) for k in merged.keys()})

    # 顺便算每个 (dataset, prompt_type) 的样本数
    samples_per = defaultdict(set)
    for key in per_sample.keys():
        ds, pt, _, _, sid = key
        samples_per[(ds, pt)].add(sid)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "segment", "layer", "expert",
            "r_fast", "r_slow", "r_default",
            "g_fast", "g_slow", "g_default",
            "delta_r", "delta_g",
            "n_tokens_fast", "n_tokens_slow", "n_tokens_default",
            "n_samples_fast", "n_samples_slow", "n_samples_default",
        ])
        for ds, seg, layer in keys:
            a_fast = merged.get((ds, "fast", seg, layer))
            a_slow = merged.get((ds, "slow", seg, layer))
            a_def = merged.get((ds, "default", seg, layer))

            r_fast = a_fast.selection_rate() if a_fast else np.zeros(N_EXPERTS)
            r_slow = a_slow.selection_rate() if a_slow else np.zeros(N_EXPERTS)
            r_def = a_def.selection_rate() if a_def else np.zeros(N_EXPERTS)
            g_fast = a_fast.mean_gate_weight() if a_fast else np.zeros(N_EXPERTS)
            g_slow = a_slow.mean_gate_weight() if a_slow else np.zeros(N_EXPERTS)
            g_def = a_def.mean_gate_weight() if a_def else np.zeros(N_EXPERTS)

            n_fast = a_fast.total_tokens if a_fast else 0
            n_slow = a_slow.total_tokens if a_slow else 0
            n_def = a_def.total_tokens if a_def else 0
            ns_fast = len(samples_per.get((ds, "fast"), set()))
            ns_slow = len(samples_per.get((ds, "slow"), set()))
            ns_def = len(samples_per.get((ds, "default"), set()))

            for e in range(N_EXPERTS):
                w.writerow([
                    ds, seg, layer, e,
                    f"{r_fast[e]:.6f}", f"{r_slow[e]:.6f}", f"{r_def[e]:.6f}",
                    f"{g_fast[e]:.6f}", f"{g_slow[e]:.6f}", f"{g_def[e]:.6f}",
                    f"{r_slow[e] - r_fast[e]:.6f}",
                    f"{g_slow[e] - g_fast[e]:.6f}",
                    n_fast, n_slow, n_def,
                    ns_fast, ns_slow, ns_def,
                ])
    print(f"[OK] 写入 {path}（{len(keys)} 个 (dataset,segment,layer) × {N_EXPERTS} expert = "
          f"{len(keys) * N_EXPERTS} 行）")


def print_sanity_report(merged, datasets):
    """打印 sanity：每段各层 max |Δr| 看看噪声底 vs 信号。"""
    print()
    print("=" * 70)
    print("Sanity 报告：max |Δr| over experts，按 (dataset, segment, layer) 浏览")
    print("（IMAGE 段应接近噪声底 < 0.005，其它段应显著更大）")
    print("=" * 70)
    for ds in datasets:
        print(f"\n----- {ds} -----")
        for seg in SEGMENT_NAMES:
            per_layer_max = []
            for l in range(N_LAYERS):
                a_fast = merged.get((ds, "fast", seg, l))
                a_slow = merged.get((ds, "slow", seg, l))
                if a_fast is None or a_slow is None:
                    per_layer_max.append(0.0)
                    continue
                d = np.abs(a_slow.selection_rate() - a_fast.selection_rate()).max()
                per_layer_max.append(float(d))
            arr = np.array(per_layer_max)
            print(f"  {seg:12s} | "
                  f"max(layers)={arr.max():.4f} (layer {int(arr.argmax()):2d}) | "
                  f"mean={arr.mean():.4f} | "
                  f"median={np.median(arr):.4f}")


if __name__ == "__main__":
    main()

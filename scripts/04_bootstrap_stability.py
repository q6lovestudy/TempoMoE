"""
SAFEx 风格的 bootstrap 稳定性选择

为什么需要：
  单次跑出来的 |Δr| 排名不稳定——换一批样本、换 prompt 写法都可能让排名翻车。
  论文里直接报"top-K 慢专家"的话，审稿人会问"你怎么知道这不是噪声？"

做法：
  1. 把样本随机打散，重复 B 轮（每轮取 50% 样本）
  2. 每轮重算每段每层的 Δr，记录每段每层 |Δr| 排前 K 的专家
  3. 累计：每个 (segment, layer, expert) 在多少轮里入选 top-K
  4. 入选率 π = 计数 / B，作为稳定性分数
  5. π ≥ 0.7 视为"稳定的"slow/fast 候选专家

输出：
  runs/pilot/analysis/stability_scores.csv
  字段：dataset, segment, layer, expert, leaning(slow/fast),
        mean_delta_r, std_delta_r, stability_pi, n_rounds_in_topk

跑法：
    python scripts/04_bootstrap_stability.py            # 默认 B=50, top_k_per_layer=10
    python scripts/04_bootstrap_stability.py --rounds 100 --top_k 8
"""

import argparse
import csv
import random
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
    p.add_argument("--rounds", type=int, default=50,
                   help="Bootstrap 轮数 B")
    p.add_argument("--subsample_frac", type=float, default=0.5,
                   help="每轮采样比例（默认 0.5）")
    p.add_argument("--top_k_per_layer", type=int, default=10,
                   help="每段每层取 |Δr| 排前 K 的专家进入 top-K 计数")
    p.add_argument("--stable_threshold", type=float, default=0.7,
                   help="π 大于等于此值视为稳定（仅影响 sanity 输出，不影响 CSV 完整性）")
    p.add_argument("--seed", type=int, default=42)
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

    print("[INFO] 第一遍聚合（per_sample，下面 bootstrap 在内存里重复合并）...")
    per_sample = aggregate_records(records, routes_dir, per_sample=True)

    # 按 dataset 分别 bootstrap
    datasets = sorted({key[0] for key in per_sample.keys()})
    print(f"[INFO] 数据集: {datasets}")
    print(f"[INFO] B={args.rounds}, subsample={args.subsample_frac}, "
          f"top_k_per_layer={args.top_k_per_layer}")

    rng = random.Random(args.seed)

    # 准备输出收集器
    # results[(dataset, segment, layer, expert)] -> {
    #     "n_topk_slow": int,    # 入选 slow-leaning top-K 的轮数
    #     "n_topk_fast": int,    # 入选 fast-leaning top-K 的轮数
    #     "deltas": list[float], # 每轮 Δr 值（用来算 mean/std）
    # }
    results = defaultdict(lambda: {
        "n_topk_slow": 0,
        "n_topk_fast": 0,
        "deltas": [],
    })

    # 每个数据集独立 bootstrap
    for ds in datasets:
        ds_sample_ids = sorted({k[4] for k in per_sample.keys() if k[0] == ds})
        n_sub = max(1, int(len(ds_sample_ids) * args.subsample_frac))
        print(f"\n[{ds}] 总 sample={len(ds_sample_ids)}, "
              f"每轮采样={n_sub}, 跑 {args.rounds} 轮 ...")

        for b in range(args.rounds):
            sub = set(rng.sample(ds_sample_ids, n_sub))
            merged = merge_per_sample(per_sample, sample_ids=sub)
            # merge_per_sample 不区分 dataset 但按 sample_id 过滤
            # 因为我们这里只挑了 ds 的 sample_ids，所以 merged 里只剩 ds 的 keys

            for seg in SEGMENT_NAMES:
                for l in range(N_LAYERS):
                    a_fast = merged.get((ds, "fast", seg, l))
                    a_slow = merged.get((ds, "slow", seg, l))
                    if a_fast is None or a_slow is None:
                        continue
                    delta = a_slow.selection_rate() - a_fast.selection_rate()

                    # 记录每个专家的 Δr 值
                    for e in range(N_EXPERTS):
                        results[(ds, seg, l, e)]["deltas"].append(float(delta[e]))

                    # 标记本轮 top-K slow-leaning
                    top_slow = np.argsort(-delta)[: args.top_k_per_layer]   # 大的 K 个
                    for e in top_slow:
                        if delta[e] > 0:
                            results[(ds, seg, l, int(e))]["n_topk_slow"] += 1

                    # top-K fast-leaning
                    top_fast = np.argsort(delta)[: args.top_k_per_layer]    # 小的 K 个
                    for e in top_fast:
                        if delta[e] < 0:
                            results[(ds, seg, l, int(e))]["n_topk_fast"] += 1

            if (b + 1) % 10 == 0:
                print(f"  ... 第 {b + 1}/{args.rounds} 轮完成")

    # ----------- 写 CSV -----------
    out_path = out_dir / "stability_scores.csv"
    write_stability_csv(results, out_path, args.rounds)

    # ----------- 打印稳定性 sanity -----------
    print_top_stable(results, args.stable_threshold, args.rounds)

    print(f"\n[OK] 写入: {out_path}")


def write_stability_csv(results, path, B):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "segment", "layer", "expert", "leaning",
            "mean_delta_r", "std_delta_r",
            "stability_pi", "n_rounds_in_topk",
            "n_rounds",
        ])
        for key, data in sorted(results.items()):
            ds, seg, layer, expert = key
            deltas = np.array(data["deltas"])
            mean_d = float(deltas.mean()) if len(deltas) else 0.0
            std_d = float(deltas.std()) if len(deltas) else 0.0

            # slow-leaning：以 top_slow 入选率为 π
            n_slow = data["n_topk_slow"]
            n_fast = data["n_topk_fast"]
            # 同一专家 slow 和 fast 入选率不会同时高（Δr 符号矛盾）
            # 哪个大用哪个，并标记 leaning
            if n_slow >= n_fast:
                leaning = "slow"
                n_in_topk = n_slow
            else:
                leaning = "fast"
                n_in_topk = n_fast
            pi = n_in_topk / B if B > 0 else 0.0

            # 没进过 top-K 的就不写（行数太多没必要）
            if n_in_topk == 0:
                continue

            w.writerow([
                ds, seg, layer, expert, leaning,
                f"{mean_d:.6f}", f"{std_d:.6f}",
                f"{pi:.4f}", n_in_topk,
                B,
            ])


def print_top_stable(results, threshold, B):
    """每段每层打印稳定 slow / fast 专家"""
    print()
    print("=" * 70)
    print(f"稳定专家清单（π ≥ {threshold}，每段每层最多列 5 个）")
    print("=" * 70)

    # 按 (dataset, segment) 分桶
    by_seg = defaultdict(list)
    for (ds, seg, layer, expert), data in results.items():
        n = max(data["n_topk_slow"], data["n_topk_fast"])
        if n / B < threshold:
            continue
        leaning = "slow" if data["n_topk_slow"] >= data["n_topk_fast"] else "fast"
        mean_d = float(np.mean(data["deltas"]))
        by_seg[(ds, seg)].append((layer, expert, leaning, mean_d, n / B))

    for (ds, seg), rows in sorted(by_seg.items()):
        print(f"\n[{ds} | {seg}] 共 {len(rows)} 个稳定专家")
        # 按层组织
        by_layer = defaultdict(list)
        for layer, expert, leaning, md, pi in rows:
            by_layer[layer].append((expert, leaning, md, pi))
        for layer in sorted(by_layer.keys())[:N_LAYERS]:
            items = sorted(by_layer[layer], key=lambda x: -abs(x[2]))[:5]
            parts = []
            for e, leaning, md, pi in items:
                tag = "S" if leaning == "slow" else "F"
                parts.append(f"e{e:02d}({tag},Δr={md:+.4f},π={pi:.2f})")
            print(f"  layer {layer:2d}: {' '.join(parts)}")


if __name__ == "__main__":
    main()

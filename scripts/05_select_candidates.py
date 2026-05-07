"""
合并 03 的 Δr 分数 + 04 的稳定性，输出最终候选 fast/slow 专家清单。

筛选规则（默认）：
  - π ≥ 0.7（在 04 的 bootstrap 里至少 70% 轮次进入 top-K）
  - |mean_delta_r| ≥ 阈值（避免选到信号太弱的）
  - 优先用 question / decode 段（image 段都是噪声底，instruction 含字面干扰）

跨数据集策略：
  - "通用慢专家"：在 RealWorldQA 和 MathVista 都被选中且方向一致 → 高置信度
  - "任务特异慢专家"：只在一个数据集被选中 → 标记，可能是任务特异

输出：
  runs/pilot/analysis/candidates.csv
    每行：dataset, segment, layer, expert, leaning,
          mean_delta_r, stability_pi, is_universal

  runs/pilot/analysis/candidates_universal.csv
    只列在两个数据集都稳定且方向一致的专家

跑法：
    python scripts/05_select_candidates.py                            # 默认
    python scripts/05_select_candidates.py --segments question decode  # 只看 question + decode
    python scripts/05_select_candidates.py --min_pi 0.6 --min_dr 0.005
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--analysis_dir", default=None,
                   help="03/04 输出所在目录，默认 runs/pilot/analysis")
    p.add_argument("--segments", nargs="+",
                   default=["question", "decode"],
                   choices=["image", "question", "instruction", "decode"],
                   help="只考虑这些段（默认 question + decode）")
    p.add_argument("--min_pi", type=float, default=0.7,
                   help="稳定性阈值")
    p.add_argument("--min_dr", type=float, default=0.005,
                   help="|mean Δr| 最小阈值（小于此值视为信号不够强）")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    analysis_dir = Path(args.analysis_dir or (PROJECT_ROOT / "runs" / "pilot" / "analysis"))
    stability_path = analysis_dir / "stability_scores.csv"
    out_all = analysis_dir / "candidates.csv"
    out_universal = analysis_dir / "candidates_universal.csv"

    if not stability_path.exists():
        raise FileNotFoundError(
            f"找不到 {stability_path}，请先跑 04_bootstrap_stability.py"
        )

    print(f"[INFO] 读 stability: {stability_path}")
    rows = []
    with open(stability_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"[INFO] 共 {len(rows)} 行")

    # 应用筛选规则
    kept = []
    for r in rows:
        if r["segment"] not in args.segments:
            continue
        pi = float(r["stability_pi"])
        if pi < args.min_pi:
            continue
        mean_dr = float(r["mean_delta_r"])
        if abs(mean_dr) < args.min_dr:
            continue
        kept.append(r)
    print(f"[INFO] 通过 π≥{args.min_pi} 且 |Δr|≥{args.min_dr} 筛选: {len(kept)} 行")

    # ----- 全部候选 -----
    write_csv(kept, out_all, include_universal=False)

    # ----- 找跨数据集一致的"通用专家" -----
    # 按 (segment, layer, expert, leaning) 分组，看在多少个 dataset 都通过
    grouped = defaultdict(list)
    for r in kept:
        key = (r["segment"], int(r["layer"]), int(r["expert"]), r["leaning"])
        grouped[key].append(r)

    datasets = sorted({r["dataset"] for r in kept})
    universal_rows = []
    for key, items in grouped.items():
        if len(items) < 2:
            continue
        ds_set = {it["dataset"] for it in items}
        if not (len(ds_set) >= len(datasets) and ds_set >= set(datasets)):
            continue
        # 在每个 dataset 都进过筛 -> universal
        seg, layer, expert, leaning = key
        # 平均 Δr 和 π
        mean_dr = sum(float(it["mean_delta_r"]) for it in items) / len(items)
        mean_pi = sum(float(it["stability_pi"]) for it in items) / len(items)
        universal_rows.append({
            "segment": seg,
            "layer": layer,
            "expert": expert,
            "leaning": leaning,
            "mean_delta_r_avg_across_datasets": f"{mean_dr:.6f}",
            "stability_pi_avg": f"{mean_pi:.4f}",
            "datasets": ",".join(sorted(ds_set)),
        })

    if universal_rows:
        with open(out_universal, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(universal_rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(
                universal_rows,
                key=lambda r: (r["segment"], r["leaning"], int(r["layer"]), -abs(float(r["mean_delta_r_avg_across_datasets"]))),
            ))
        print(f"[OK] 通用候选写入: {out_universal} ({len(universal_rows)} 行)")
    else:
        print("[WARN] 没有跨数据集一致的通用候选；可能数据集差异大，或阈值太严。"
              "建议放宽 --min_pi 或 --min_dr 再试，或单独看每个数据集。")

    # ----- 总结打印 -----
    print()
    print("=" * 60)
    print(f"候选专家清单总结（segments={args.segments}, min_pi={args.min_pi}, min_dr={args.min_dr}）")
    print("=" * 60)
    by_ds_seg = defaultdict(lambda: {"slow": [], "fast": []})
    for r in kept:
        by_ds_seg[(r["dataset"], r["segment"])][r["leaning"]].append(r)
    for (ds, seg), groups in sorted(by_ds_seg.items()):
        n_slow = len(groups["slow"])
        n_fast = len(groups["fast"])
        print(f"  [{ds} | {seg:11s}]  slow候选 {n_slow:3d}  |  fast候选 {n_fast:3d}")
    print()
    print(f"通用候选（两数据集都过筛 + 方向一致）: {len(universal_rows)} 个")


def write_csv(rows, path, include_universal):
    """写候选 CSV，按 (dataset, segment, leaning, layer, |Δr|) 排序。"""
    if not rows:
        print(f"[WARN] 空候选，跳过 {path}")
        return
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["dataset"], r["segment"], r["leaning"],
            int(r["layer"]),
            -abs(float(r["mean_delta_r"])),
        ),
    )
    fields = [
        "dataset", "segment", "layer", "expert", "leaning",
        "mean_delta_r", "std_delta_r", "stability_pi", "n_rounds_in_topk", "n_rounds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[OK] 全部候选写入: {path} ({len(rows_sorted)} 行)")


if __name__ == "__main__":
    main()

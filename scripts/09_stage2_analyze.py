"""
Stage 2 分析：对比 baseline (Stage 1 跑的 main_records) vs 各干预实验。

核心指标（按 dataset × prompt_type 分桶比较）：
    - mean_output_tokens         : 输出长度 → 是否变短/变长
    - mean_think_tokens          : think 段长度 → 慢思考被抑制了吗
    - rate_has_think_marker      : 有 think 标签的比例
    - rate_simple_match_correct  : 简单字符匹配准确率（粗估）

输出：
    runs/stage2/comparison.csv     表格对比
    runs/stage2/figures/           对比柱状图（按指标）

跑法：
    python scripts/09_stage2_analyze.py
    python scripts/09_stage2_analyze.py --baseline runs/pilot/main_records.jsonl
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 中文字体
_CANDIDATE_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
for _fp in _CANDIDATE_FONT_FILES:
    try:
        _fm.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=str(
        PROJECT_ROOT / "runs" / "pilot" / "main_records.jsonl"))
    p.add_argument("--stage2_root", default=str(
        PROJECT_ROOT / "runs" / "stage2"))
    p.add_argument("--out_dir", default=str(
        PROJECT_ROOT / "runs" / "stage2"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------ 加载 baseline ------
    print(f"[INFO] baseline: {args.baseline}")
    baseline_recs = _load_jsonl(args.baseline)
    print(f"[INFO] baseline 共 {len(baseline_recs)} 条")

    # ------ 加载所有 stage2 实验 ------
    stage2_root = Path(args.stage2_root)
    experiments = {}
    for sub in sorted(stage2_root.glob("*")):
        if not sub.is_dir():
            continue
        jsonl = sub / "main_records.jsonl"
        if not jsonl.exists():
            continue
        recs = _load_jsonl(str(jsonl))
        if recs:
            experiments[sub.name] = recs
            print(f"[INFO] 实验 {sub.name}: {len(recs)} 条")

    if not experiments:
        raise SystemExit(f"没找到 Stage 2 实验数据，请先跑 08_stage2_run.py")

    # ------ Match-sample 对比：把 baseline 筛到与 stage2 同样的 (sample, variant) 子集 ------
    # 否则 smoke 时 baseline=50样本均值 vs stage2=3样本均值，根本不是同一组样本，比不了
    stage2_keys = set()
    for recs in experiments.values():
        for r in recs:
            stage2_keys.add((r["sample_id"], r["variant_id"]))

    matched_baseline = [
        r for r in baseline_recs
        if (r["sample_id"], r["variant_id"]) in stage2_keys
    ]
    print(f"[INFO] 全 baseline {len(baseline_recs)} 条；与 stage2 匹配的子集 {len(matched_baseline)} 条")
    print(f"[INFO] 对比将基于这 {len(matched_baseline)} 条 baseline（保证苹果对苹果）")

    # ------ 算指标 ------
    # 结构：metrics[exp_name][(dataset, prompt_type)] = dict of metric values
    print("\n[INFO] 计算各组指标 ...")
    all_metrics = {"baseline": compute_metrics(matched_baseline)}
    for exp_name, recs in experiments.items():
        all_metrics[exp_name] = compute_metrics(recs)

    # ------ 写对比 CSV ------
    csv_path = out_dir / "comparison.csv"
    write_comparison_csv(all_metrics, csv_path)

    # ------ 出图 ------
    plot_comparison_bars(all_metrics, fig_dir)

    # ------ 终端打印关键对比 ------
    print_key_comparison(all_metrics)


# ----------------------------------------------------------------------
def _load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_metrics(records):
    """按 (dataset, prompt_type) 分桶算指标。"""
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["dataset"], r["prompt_type"])].append(r)

    out = {}
    for key, recs in grouped.items():
        out_lens = [r["output_token_count"] for r in recs]
        # think 段长度：think_token_range 的 [start, end)
        think_lens = []
        for r in recs:
            tr = r.get("think_token_range")
            if tr and isinstance(tr, list) and len(tr) == 2:
                think_lens.append(tr[1] - tr[0])
            else:
                think_lens.append(0)
        rate_think = sum(1 for r in recs if r.get("has_think_marker")) / len(recs)

        # 简单字符匹配准确率（粗估）
        rate_correct = compute_simple_match_accuracy(recs)

        out[key] = {
            "n": len(recs),
            "mean_output_tokens": float(np.mean(out_lens)),
            "median_output_tokens": float(np.median(out_lens)),
            "mean_think_tokens": float(np.mean(think_lens)),
            "rate_has_think_marker": rate_think,
            "rate_simple_match_correct": rate_correct,
        }
    return out


_NORM_RE = re.compile(r"[^a-z0-9]+")

def _normalize_answer(s: str) -> str:
    """把答案做粗暴归一：小写、去标点空格。"""
    if s is None:
        return ""
    return _NORM_RE.sub("", s.lower().strip())


def compute_simple_match_accuracy(records):
    """非常粗的字符串匹配准确率：parsed_answer 包含 ground_truth 即算对。
    适用于数字/单词类答案；多选题、长文本会偏低，但作为 trend 比较够用。"""
    n_correct = 0
    n_eligible = 0
    for r in records:
        gt = _normalize_answer(r.get("ground_truth", ""))
        pred = _normalize_answer(r.get("parsed_answer", ""))
        if not gt:
            continue
        n_eligible += 1
        if gt and pred and (gt in pred or pred in gt):
            n_correct += 1
    return n_correct / n_eligible if n_eligible > 0 else 0.0


# ----------------------------------------------------------------------
def write_comparison_csv(all_metrics, path):
    """写 long-format CSV，便于 pivot。"""
    rows = []
    for exp_name, metrics in all_metrics.items():
        for (ds, pt), vals in metrics.items():
            row = {
                "experiment": exp_name,
                "dataset": ds,
                "prompt_type": pt,
                **vals,
            }
            rows.append(row)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] 写入 {path}")


def print_key_comparison(all_metrics):
    """打印 baseline vs 各实验的关键 delta。"""
    print()
    print("=" * 90)
    print("关键指标对比（vs baseline）")
    print("=" * 90)
    if "baseline" not in all_metrics:
        print("[WARN] 没找到 baseline，跳过")
        return

    baseline = all_metrics["baseline"]
    for exp_name in [e for e in all_metrics if e != "baseline"]:
        metrics = all_metrics[exp_name]
        print(f"\n--- {exp_name} ---")
        print(f"{'数据集':<14} {'prompt':<8} | "
              f"{'输出长度':>14} {'think长度':>14} {'有think%':>10} {'匹配率%':>10}")
        for key in sorted(metrics.keys()):
            ds, pt = key
            if key not in baseline:
                continue
            b = baseline[key]
            m = metrics[key]
            d_out = m["mean_output_tokens"] - b["mean_output_tokens"]
            d_think = m["mean_think_tokens"] - b["mean_think_tokens"]
            d_rate = (m["rate_has_think_marker"] - b["rate_has_think_marker"]) * 100
            d_acc = (m["rate_simple_match_correct"] - b["rate_simple_match_correct"]) * 100
            print(f"{ds:<14} {pt:<8} | "
                  f"{m['mean_output_tokens']:>6.1f} ({d_out:+5.1f}) "
                  f"{m['mean_think_tokens']:>6.1f} ({d_think:+5.1f}) "
                  f"{m['rate_has_think_marker']*100:>5.1f} ({d_rate:+4.1f}) "
                  f"{m['rate_simple_match_correct']*100:>5.1f} ({d_acc:+4.1f})")


# ----------------------------------------------------------------------
def plot_comparison_bars(all_metrics, fig_dir):
    """画 baseline vs 各实验的关键指标柱状对比，分 dataset。"""
    metric_keys = [
        ("mean_output_tokens",       "平均输出长度（tokens）"),
        ("mean_think_tokens",        "平均 think 段长度（tokens）"),
        ("rate_has_think_marker",    "有 think 标签的比例"),
        ("rate_simple_match_correct","简单匹配准确率"),
    ]
    exp_names = list(all_metrics.keys())
    if "baseline" in exp_names:
        # baseline 总放第一个
        exp_names.remove("baseline")
        exp_names = ["baseline"] + exp_names

    # 收集所有 (dataset, prompt_type) 组合
    all_keys = set()
    for m in all_metrics.values():
        all_keys.update(m.keys())
    datasets = sorted({k[0] for k in all_keys})
    prompt_types = ["fast", "slow", "default"]

    for metric_key, metric_label in metric_keys:
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(len(prompt_types))
            width = 0.8 / len(exp_names)

            for i, exp in enumerate(exp_names):
                vals = []
                for pt in prompt_types:
                    v = all_metrics[exp].get((ds, pt), {}).get(metric_key)
                    vals.append(v if v is not None else 0.0)
                ax.bar(x + i*width, vals, width, label=exp,
                       alpha=0.85 if exp != "baseline" else 1.0,
                       edgecolor="black", linewidth=0.5)

            ax.set_xticks(x + width * (len(exp_names)-1) / 2)
            ax.set_xticklabels(prompt_types)
            ax.set_xlabel("prompt_type")
            ax.set_ylabel(metric_label)
            ax.set_title(f"[{ds}] {metric_label}\nbaseline vs Stage 2 干预实验")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            out_path = fig_dir / f"compare_{metric_key}_{ds}.png"
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()

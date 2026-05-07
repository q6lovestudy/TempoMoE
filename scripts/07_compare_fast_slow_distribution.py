"""
画快慢专家候选分布对比图。

读 candidates_universal.csv（跨数据集稳定的 234 个候选），
出 1 张 4 子图的对比图，把快/慢专家分布对照清楚：

  ┌──────────────────────────┬──────────────────────────┐
  │ A. 总览柱状图             │ B. 按层分布               │
  │   按 segment×leaning 分桶 │   每层快/慢候选数对比      │
  ├──────────────────────────┼──────────────────────────┤
  │ C. Layer×Expert 候选矩阵  │ D. 按 |Δr| 强度的散点      │
  │   红 = slow, 蓝 = fast    │   x=layer, y=Δr, 色=segment│
  └──────────────────────────┴──────────────────────────┘

输出: runs/pilot/figures/fig_fast_slow_distribution.png
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------- 中文字体配置（与 06 一致） ----------
_CANDIDATE_FONT_FILES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
for _fp in _CANDIDATE_FONT_FILES:
    try:
        _fm.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei", "Source Han Sans CN", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


N_LAYERS = 26
N_EXPERTS = 64
SLOW_COLOR = "#d62728"   # 红 = slow
FAST_COLOR = "#1f77b4"   # 蓝 = fast


def load_universal(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["layer"] = int(r["layer"])
            r["expert"] = int(r["expert"])
            r["mean_delta_r"] = float(r["mean_delta_r_avg_across_datasets"])
            r["pi"] = float(r["stability_pi_avg"])
            rows.append(r)
    return rows


def main():
    csv_path = PROJECT_ROOT / "runs" / "pilot" / "analysis" / "candidates_universal.csv"
    out_path = PROJECT_ROOT / "runs" / "pilot" / "figures" / "fig_fast_slow_distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_universal(csv_path)
    n_total = len(rows)
    print(f"[INFO] 读 {csv_path.name}: 共 {n_total} 个 universal 候选")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Fast vs Slow 专家候选分布（共 {n_total} 个跨数据集稳定候选）",
        fontsize=15, fontweight="bold",
    )

    # ============ A: 总览柱状图 ============
    ax = axes[0][0]
    plot_summary_bars(ax, rows)

    # ============ B: 按层分布 ============
    ax = axes[0][1]
    plot_per_layer_bars(ax, rows)

    # ============ C: Layer × Expert 候选矩阵 ============
    ax = axes[1][0]
    plot_layer_expert_grid(ax, rows)

    # ============ D: |Δr| 强度散点 ============
    ax = axes[1][1]
    plot_delta_r_scatter(ax, rows)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] 写入 {out_path}")


# ----------------------------------------------------------------------
# 子图 A: 总览
# ----------------------------------------------------------------------
def plot_summary_bars(ax, rows):
    """按 (segment, leaning) 分桶画柱状图。"""
    counter = Counter()
    for r in rows:
        counter[(r["segment"], r["leaning"])] += 1

    segments = ["question", "decode"]
    x = np.arange(len(segments))
    width = 0.35

    fast_counts = [counter[(s, "fast")] for s in segments]
    slow_counts = [counter[(s, "slow")] for s in segments]

    bars_fast = ax.bar(x - width/2, fast_counts, width, label="fast 候选", color=FAST_COLOR)
    bars_slow = ax.bar(x + width/2, slow_counts, width, label="slow 候选", color=SLOW_COLOR)

    for bars in [bars_fast, bars_slow]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1, f"{int(h)}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(segments)
    ax.set_xlabel("Token 段")
    ax.set_ylabel("候选专家数")
    ax.set_title("A. 总览：每段的 fast/slow 候选数量")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


# ----------------------------------------------------------------------
# 子图 B: 每层快/慢分布
# ----------------------------------------------------------------------
def plot_per_layer_bars(ax, rows):
    """每层 fast/slow 候选数（合并 question + decode 段）。"""
    fast_per_layer = np.zeros(N_LAYERS, dtype=int)
    slow_per_layer = np.zeros(N_LAYERS, dtype=int)
    for r in rows:
        if r["leaning"] == "fast":
            fast_per_layer[r["layer"]] += 1
        else:
            slow_per_layer[r["layer"]] += 1

    x = np.arange(N_LAYERS)
    width = 0.4
    ax.bar(x - width/2, fast_per_layer, width, label="fast 候选", color=FAST_COLOR)
    ax.bar(x + width/2, slow_per_layer, width, label="slow 候选", color=SLOW_COLOR)

    ax.set_xlabel("MoE Layer (0..25)")
    ax.set_ylabel("候选数（含 question + decode 段）")
    ax.set_title("B. 每层快/慢候选数对比")
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


# ----------------------------------------------------------------------
# 子图 C: Layer × Expert 候选矩阵
# ----------------------------------------------------------------------
def plot_layer_expert_grid(ax, rows):
    """layer (y) × expert id (x) 网格，红点=slow，蓝点=fast，
    点大小代表 |Δr| 强度，alpha 代表 stability π。"""
    for r in rows:
        color = SLOW_COLOR if r["leaning"] == "slow" else FAST_COLOR
        size = max(20, abs(r["mean_delta_r"]) * 1500)   # 放大便于看到
        ax.scatter(
            r["expert"], r["layer"],
            s=size, color=color, alpha=min(1.0, r["pi"]),
            edgecolors="black", linewidths=0.3,
            marker="s",
        )

    ax.set_xlabel("Expert id (0..63)")
    ax.set_ylabel("MoE Layer (0..25)")
    ax.set_title("C. Layer × Expert 候选矩阵（点大小=|Δr|，透明度=π）")
    ax.set_xlim(-1, N_EXPERTS)
    ax.set_ylim(-1, N_LAYERS)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)

    # 图例
    legend_elements = [
        plt.scatter([], [], s=80, color=SLOW_COLOR, edgecolors="black",
                    linewidths=0.3, marker="s", label="slow 候选"),
        plt.scatter([], [], s=80, color=FAST_COLOR, edgecolors="black",
                    linewidths=0.3, marker="s", label="fast 候选"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


# ----------------------------------------------------------------------
# 子图 D: Δr 强度散点
# ----------------------------------------------------------------------
def plot_delta_r_scatter(ax, rows):
    """每个候选画一个点：x=layer, y=Δr, 色=segment, 标记=leaning。
    一眼看清"信号强度随层的分布"以及 question/decode 两段的关系。"""
    seg_color = {"question": "#2ca02c", "decode": "#9467bd"}

    for r in rows:
        marker = "^" if r["leaning"] == "slow" else "v"
        ax.scatter(
            r["layer"], r["mean_delta_r"],
            color=seg_color.get(r["segment"], "gray"),
            marker=marker, s=40, alpha=0.7, edgecolors="black", linewidths=0.3,
        )

    # 0 线
    ax.axhline(0, color="black", linewidth=0.5)
    # 噪声底参考线
    ax.axhspan(-0.005, 0.005, color="gray", alpha=0.15, label="噪声底 ±0.005")

    ax.set_xlabel("MoE Layer (0..25)")
    ax.set_ylabel("mean Δr (slow - fast)")
    ax.set_title("D. 候选专家的 Δr 强度（▲slow / ▼fast）")
    ax.grid(True, alpha=0.3)

    # 图例
    legend_elements = [
        plt.scatter([], [], color="#2ca02c", marker="^", s=60, label="question / slow"),
        plt.scatter([], [], color="#2ca02c", marker="v", s=60, label="question / fast"),
        plt.scatter([], [], color="#9467bd", marker="^", s=60, label="decode / slow"),
        plt.scatter([], [], color="#9467bd", marker="v", s=60, label="decode / fast"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)


if __name__ == "__main__":
    main()

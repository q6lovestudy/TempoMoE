"""
出图脚本，全部 PNG 保存到 runs/pilot/figures/。

5 张关键图：
  fig1_layerwise_max_dr.png       每段 max|Δr| 随层变化（确认信号在哪些层最强）
  fig2_dr_heatmap.png             Δr 热图 [layer × expert]，每段一个子图
  fig3_top_candidates_bar.png     选中的 top 候选专家柱状图
  fig4_output_length_dist.png     fast/slow/default 输出长度分布（行为侧 sanity）
  fig5_stability_distribution.png 稳定性分布直方图（看大部分专家 π 在哪个区间）

跑法：
    python scripts/06_visualize.py
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# ---------- 中文字体配置 ----------
# 系统里如果有 Noto CJK / 文泉驿，先注册一下保证 matplotlib 能找到
# 再设 rcParams 的字体优先级：找到哪个就用哪个，最后兜底 DejaVu
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

# 优先级列表：matplotlib 会按顺序找第一个可用的
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",      # 简体优先
    "Noto Sans CJK JP",      # JP 字体也能正确渲染汉字（CJK 共享）
    "WenQuanYi Zen Hei",     # 文泉驿正黑
    "WenQuanYi Micro Hei",
    "Source Han Sans CN",
    "SimHei",                # Windows 黑体（一般 Linux 没有）
    "DejaVu Sans",           # 兜底，不支持中文，但保证不崩
]
# 防止负号被显示为方框
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import N_LAYERS, N_EXPERTS, SEGMENT_NAMES


SEGMENT_COLORS = {
    "image":       "#888888",
    "question":    "#1f77b4",
    "instruction": "#ff7f0e",
    "decode":      "#d62728",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--analysis_dir", default=None)
    p.add_argument("--figures_dir", default=None)
    p.add_argument("--dataset", default=None,
                   help="只画指定数据集，None 则每个数据集都画（一份图 per dataset）")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main_records_path = cfg["paths"]["main_records"]
    analysis_dir = Path(args.analysis_dir or (PROJECT_ROOT / "runs" / "pilot" / "analysis"))
    figures_dir = Path(args.figures_dir or (PROJECT_ROOT / "runs" / "pilot" / "figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 读 expert scores
    score_csv = analysis_dir / "expert_scores.csv"
    if not score_csv.exists():
        raise FileNotFoundError(f"找不到 {score_csv}，请先跑 03_compute_expert_scores.py")
    print(f"[INFO] 读 {score_csv}")
    scores = load_scores(score_csv)

    # 读 stability scores
    stab_csv = analysis_dir / "stability_scores.csv"
    stability = load_stability(stab_csv) if stab_csv.exists() else {}
    if not stability:
        print(f"[WARN] 没有 stability_scores.csv，fig5 会跳过；建议先跑 04_bootstrap_stability.py")

    # 读 main_records 用于 fig4 输出长度
    records = []
    with open(main_records_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 决定要画哪些数据集
    datasets_in_data = sorted({r["dataset"] for r in records})
    target_datasets = [args.dataset] if args.dataset else datasets_in_data

    # ----- fig4 用所有 records，不分数据集 -----
    plot_output_length(records, figures_dir / "fig4_output_length_dist.png")

    # ----- fig1, fig2, fig3, fig5 按数据集画 -----
    for ds in target_datasets:
        ds_scores = [r for r in scores if r["dataset"] == ds]
        if not ds_scores:
            print(f"[WARN] {ds} 没有 scores 数据，跳过")
            continue

        plot_layerwise_max_dr(
            ds_scores, ds,
            figures_dir / f"fig1_layerwise_max_dr_{ds}.png",
        )
        plot_dr_heatmap(
            ds_scores, ds,
            figures_dir / f"fig2_dr_heatmap_{ds}.png",
        )

        if stability:
            ds_stab = [r for r in stability if r["dataset"] == ds]
            plot_top_candidates_bar(
                ds_stab, ds,
                figures_dir / f"fig3_top_candidates_bar_{ds}.png",
            )
            plot_stability_distribution(
                ds_stab, ds,
                figures_dir / f"fig5_stability_distribution_{ds}.png",
            )

    print(f"\n[OK] 全部图已保存到: {figures_dir}")


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def load_scores(path):
    """读 expert_scores.csv，返回 list of dicts（数值字段已转 float/int）。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ["layer", "expert", "n_tokens_fast", "n_tokens_slow", "n_tokens_default",
                      "n_samples_fast", "n_samples_slow", "n_samples_default"]:
                r[k] = int(r[k]) if r[k] else 0
            for k in ["r_fast", "r_slow", "r_default", "g_fast", "g_slow", "g_default",
                      "delta_r", "delta_g"]:
                r[k] = float(r[k]) if r[k] else 0.0
            out.append(r)
    return out


def load_stability(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ["layer", "expert", "n_rounds_in_topk", "n_rounds"]:
                r[k] = int(r[k]) if r[k] else 0
            for k in ["mean_delta_r", "std_delta_r", "stability_pi"]:
                r[k] = float(r[k]) if r[k] else 0.0
            out.append(r)
    return out


# ----------------------------------------------------------------------
# fig1: layer-wise max|Δr|
# ----------------------------------------------------------------------
def plot_layerwise_max_dr(scores, dataset, out_path):
    """每段一条曲线：x=layer, y=max|Δr| over experts。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for seg in SEGMENT_NAMES:
        per_layer = np.zeros(N_LAYERS)
        for r in scores:
            if r["segment"] != seg:
                continue
            l, dr = r["layer"], abs(r["delta_r"])
            if dr > per_layer[l]:
                per_layer[l] = dr
        ax.plot(
            range(N_LAYERS), per_layer,
            label=seg, color=SEGMENT_COLORS[seg], linewidth=2, marker="o", markersize=4,
        )

    ax.set_xlabel("Layer (MoE index)")
    ax.set_ylabel("max |Δr| over 64 experts")
    ax.set_title(f"[{dataset}] Layer-wise max |Δr| per segment\n"
                 f"image 段是噪声底；其它段越显著越说明 fast/slow 信号强")
    ax.legend(title="Segment")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")


# ----------------------------------------------------------------------
# fig2: Δr 热图 [layer × expert]，4 个子图（每段一个）
# ----------------------------------------------------------------------
def plot_dr_heatmap(scores, dataset, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    # 全局色阶（用所有段最大绝对值，便于对比）
    all_dr = [abs(r["delta_r"]) for r in scores if r["segment"] != "image"]
    vmax = max(all_dr) if all_dr else 0.05
    vmin = -vmax

    for i, seg in enumerate(SEGMENT_NAMES):
        ax = axes[i]
        mat = np.zeros((N_LAYERS, N_EXPERTS))
        for r in scores:
            if r["segment"] != seg:
                continue
            mat[r["layer"], r["expert"]] = r["delta_r"]
        im = ax.imshow(
            mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{seg}")
        ax.set_xlabel("expert id (0..63)")
        ax.set_ylabel("layer (0..25)")
        plt.colorbar(im, ax=ax, label="Δr (slow - fast)")

    fig.suptitle(
        f"[{dataset}] Δr heatmap by segment\n"
        f"红色 = slow-leaning，蓝色 = fast-leaning",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")


# ----------------------------------------------------------------------
# fig3: 稳定 top 候选专家柱状图
# ----------------------------------------------------------------------
def plot_top_candidates_bar(stability, dataset, out_path, top_n=15, segments=("question", "decode")):
    """每段画一组，分别 slow / fast 候选 top_n。"""
    fig, axes = plt.subplots(len(segments), 2, figsize=(16, 4 * len(segments)))
    if len(segments) == 1:
        axes = np.array([axes])

    for row, seg in enumerate(segments):
        slow = sorted(
            [r for r in stability if r["segment"] == seg and r["leaning"] == "slow"],
            key=lambda r: -r["stability_pi"],
        )[:top_n]
        fast = sorted(
            [r for r in stability if r["segment"] == seg and r["leaning"] == "fast"],
            key=lambda r: -r["stability_pi"],
        )[:top_n]

        # slow 子图
        ax = axes[row][0]
        labels = [f"L{r['layer']:02d}-e{r['expert']:02d}" for r in slow]
        pis = [r["stability_pi"] for r in slow]
        drs = [r["mean_delta_r"] for r in slow]
        ax.barh(range(len(slow)), pis, color="#d62728", alpha=0.7)
        ax.set_yticks(range(len(slow)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("stability π")
        ax.set_title(f"{seg} | top {top_n} SLOW candidates by π")
        for i, (pi, dr) in enumerate(zip(pis, drs)):
            ax.text(pi + 0.005, i, f"Δr={dr:+.4f}", va="center", fontsize=7)
        ax.set_xlim(0, 1.1)

        # fast 子图
        ax = axes[row][1]
        labels = [f"L{r['layer']:02d}-e{r['expert']:02d}" for r in fast]
        pis = [r["stability_pi"] for r in fast]
        drs = [r["mean_delta_r"] for r in fast]
        ax.barh(range(len(fast)), pis, color="#1f77b4", alpha=0.7)
        ax.set_yticks(range(len(fast)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("stability π")
        ax.set_title(f"{seg} | top {top_n} FAST candidates by π")
        for i, (pi, dr) in enumerate(zip(pis, drs)):
            ax.text(pi + 0.005, i, f"Δr={dr:+.4f}", va="center", fontsize=7)
        ax.set_xlim(0, 1.1)

    fig.suptitle(f"[{dataset}] 稳定 fast/slow 候选专家", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")


# ----------------------------------------------------------------------
# fig4: 输出长度分布
# ----------------------------------------------------------------------
def plot_output_length(records, out_path):
    """fast / slow / default 的 output_token_count 直方图。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    by_type = defaultdict(list)
    for r in records:
        by_type[r["prompt_type"]].append(r["output_token_count"])

    bins = np.linspace(0, max(max(v) for v in by_type.values()) * 1.05, 50)
    colors = {"fast": "#1f77b4", "slow": "#d62728", "default": "#888888"}
    for pt in ["fast", "slow", "default"]:
        if pt not in by_type:
            continue
        vals = by_type[pt]
        ax.hist(
            vals, bins=bins, alpha=0.5, label=f"{pt} (n={len(vals)}, mean={np.mean(vals):.0f})",
            color=colors[pt],
        )
        ax.axvline(np.mean(vals), color=colors[pt], linestyle="--", linewidth=1)

    ax.set_xlabel("output_token_count")
    ax.set_ylabel("count")
    ax.set_title("Output 长度分布（按 prompt_type）\n虚线为平均值；slow > fast 才说明 prompt 起作用")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")


# ----------------------------------------------------------------------
# fig5: 稳定性 π 的分布
# ----------------------------------------------------------------------
def plot_stability_distribution(stability, dataset, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按 segment 一条直方图
    for seg in SEGMENT_NAMES:
        pis = [r["stability_pi"] for r in stability if r["segment"] == seg]
        if not pis:
            continue
        axes[0].hist(
            pis, bins=20, alpha=0.5, label=seg, color=SEGMENT_COLORS[seg],
        )
    axes[0].set_xlabel("stability π")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"[{dataset}] π 分布（按段）")
    axes[0].axvline(0.7, color="black", linestyle="--", linewidth=1, label="π=0.7 阈值")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mean Δr vs π 散点
    for seg in SEGMENT_NAMES:
        pis = [r["stability_pi"] for r in stability if r["segment"] == seg]
        drs = [r["mean_delta_r"] for r in stability if r["segment"] == seg]
        if not pis:
            continue
        axes[1].scatter(
            drs, pis, alpha=0.5, label=seg, color=SEGMENT_COLORS[seg], s=8,
        )
    axes[1].set_xlabel("mean Δr (slow - fast)")
    axes[1].set_ylabel("stability π")
    axes[1].set_title(f"[{dataset}] mean Δr vs π")
    axes[1].axhline(0.7, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(0, color="gray", linewidth=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()

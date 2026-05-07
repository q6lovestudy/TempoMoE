"""
Stage 2 因果验证主脚本：用 ExpertSuppressor 对候选专家做 mask，跑同一批样本，
                        看输出长度 / accuracy / think 段长度的变化。

跑法：
    # 跑全部预设实验
    python scripts/08_stage2_run.py

    # 只跑 1 个实验
    python scripts/08_stage2_run.py --only mask_top30_slow

    # smoke 模式：每个实验只跑 3 条样本，验证 hook 工作
    python scripts/08_stage2_run.py --smoke

预设实验（修改下面 EXPERIMENTS 字典添加新实验）：
    1. mask_top30_slow      — 抑制 top-30 慢专家（按 score = π × |Δr| 排）
    2. mask_top30_fast      — 抑制 top-30 快专家
    3. mask_random_30       — 抑制 30 个随机非候选专家（null baseline）
    4. mask_top60_slow      — 抑制 top-60 慢专家（看更激进抑制的效应）

输出：
    runs/stage2/{experiment_name}/main_records.jsonl
    每条记录格式同 Stage 1，但多一个 'experiment_name' 字段
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Set

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# 实验配置
# ----------------------------------------------------------------------
def build_experiment_configs(
    candidates_csv: str,
    n_routed_experts: int = 64,
) -> Dict[str, Dict]:
    """
    构造预设实验配置。
    每个 entry: {
        "intervention": "suppress" / "boost" / "none",
        "factor": float,
        "layer_to_experts": {layer_idx → set(expert_id)},
        "description": "...",
    }
    """
    from src.moe_intervention import (
        load_candidates_to_layer_dict, make_random_baseline, summarize_layer_dict,
    )

    # 注意：'baseline' 实验做 'none' 干预，用来对照（其实就是 Stage 1 的结果）
    # 但为了在同样代码下跑、便于对比，这里也跑一遍 baseline
    configs = {}

    # --- 1. mask top-30 slow ---
    slow_top30 = load_candidates_to_layer_dict(
        candidates_csv, leaning="slow",
        segments=["question", "decode"], top_n_by_score=30,
    )
    configs["mask_top30_slow"] = {
        "intervention": "suppress",
        "factor": 1.0,
        "layer_to_experts": slow_top30,
        "description": f"Hard mask top-30 slow 候选。{summarize_layer_dict(slow_top30)}",
    }

    # --- 2. mask top-30 fast ---
    fast_top30 = load_candidates_to_layer_dict(
        candidates_csv, leaning="fast",
        segments=["question", "decode"], top_n_by_score=30,
    )
    configs["mask_top30_fast"] = {
        "intervention": "suppress",
        "factor": 1.0,
        "layer_to_experts": fast_top30,
        "description": f"Hard mask top-30 fast 候选。{summarize_layer_dict(fast_top30)}",
    }

    # --- 3. random null baseline (匹配 slow_top30 的规模) ---
    # 排除所有 universal 候选（slow + fast），从剩下的随机抽
    all_candidates = load_candidates_to_layer_dict(
        candidates_csv, leaning=None,
        segments=["question", "decode"], top_n_by_score=None,
    )
    random_baseline_30 = make_random_baseline(
        slow_top30,
        n_routed_experts=n_routed_experts,
        seed=42,
        exclude_layer_to_experts=all_candidates,
    )
    configs["mask_random_30"] = {
        "intervention": "suppress",
        "factor": 1.0,
        "layer_to_experts": random_baseline_30,
        "description": f"Null baseline: hard mask 30 个随机非候选。{summarize_layer_dict(random_baseline_30)}",
    }

    # --- 4. mask top-60 slow（更激进，看效应是否随规模线性增） ---
    slow_top60 = load_candidates_to_layer_dict(
        candidates_csv, leaning="slow",
        segments=["question", "decode"], top_n_by_score=60,
    )
    configs["mask_top60_slow"] = {
        "intervention": "suppress",
        "factor": 1.0,
        "layer_to_experts": slow_top60,
        "description": f"Hard mask top-60 slow 候选（更激进）。{summarize_layer_dict(slow_top60)}",
    }

    return configs


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--candidates_csv", default=None,
                   help="universal 候选 CSV，默认 runs/pilot/analysis/candidates_universal.csv")
    p.add_argument("--out_root", default=None,
                   help="实验输出根目录，默认 runs/stage2")
    p.add_argument("--only", default=None,
                   help="只跑指定实验（实验名）")
    p.add_argument("--max_samples", type=int, default=None,
                   help="每个实验最多跑多少条样本（用于快速验证）")
    p.add_argument("--smoke", action="store_true",
                   help="smoke 模式：每实验只跑 3 条样本")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("stage2_run")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 路径
    candidates_csv = args.candidates_csv or str(
        PROJECT_ROOT / "runs" / "pilot" / "analysis" / "candidates_universal.csv"
    )
    out_root = Path(args.out_root or (PROJECT_ROOT / "runs" / "stage2"))
    out_root.mkdir(parents=True, exist_ok=True)

    # GPU
    from src.model_loader import (
        setup_visible_gpus, print_gpu_status,
        assert_enough_free_memory, load_model,
    )
    visible = cfg["gpu"]["visible_devices"]
    print_gpu_status(visible)
    assert_enough_free_memory(visible, cfg["gpu"].get("min_free_gib_per_gpu", 25))
    setup_visible_gpus(visible)

    # 模型
    model, processor, model_config = load_model(
        cfg["paths"]["model_dir"],
        cfg["gpu"]["max_memory_per_gpu"],
    )

    # 候选实验
    experiments = build_experiment_configs(
        candidates_csv,
        n_routed_experts=model_config.text_config.n_routed_experts,
    )
    if args.only:
        if args.only not in experiments:
            raise SystemExit(f"未知实验 '{args.only}'。可选：{list(experiments.keys())}")
        experiments = {args.only: experiments[args.only]}

    # 数据
    from src.data_loader import load_pilot_index
    samples = load_pilot_index(cfg["paths"]["pilot_index"])
    log.info(f"pilot 样本数: {len(samples)}")
    if args.smoke:
        max_samples = 3
    else:
        max_samples = args.max_samples
    if max_samples:
        samples = samples[:max_samples]
        log.info(f"按 max_samples 截断: {len(samples)} 条")

    # prompts
    from src.prompts import load_prompts
    variants = load_prompts(cfg["paths"]["prompts_yaml"])

    # 推理配置
    inference_cfg = dict(cfg["inference"])

    # 干预模块
    from src.moe_intervention import ExpertSuppressor
    from src.moe_hook import MoERouteRecorder
    from src.inference import run_one_sample
    from src.storage import save_inference_result

    # ----- 跑每个实验 -----
    for exp_name, exp_cfg in experiments.items():
        log.info("=" * 70)
        log.info(f"开始实验: {exp_name}")
        log.info(f"描述: {exp_cfg['description']}")
        log.info("=" * 70)

        # 输出目录
        exp_dir = out_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        main_records_path = str(exp_dir / "main_records.jsonl")
        routes_dir = str(exp_dir / "routes")

        # 写实验元数据，方便后续分析
        meta = {
            "experiment_name": exp_name,
            "description": exp_cfg["description"],
            "intervention": exp_cfg["intervention"],
            "factor": exp_cfg["factor"],
            "layer_to_experts": {
                str(k): sorted(list(v)) for k, v in exp_cfg["layer_to_experts"].items()
            },
            "n_samples": len(samples),
            "n_variants": len(variants),
        }
        with open(exp_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 断点续跑
        done_set = _load_done_set(main_records_path)
        log.info(f"已完成 (sample, variant): {len(done_set)} 对")

        # 装干预 + 路由 hook（路由 hook 这里也保留，便于后续验证 mask 真的生效了）
        suppressor = None
        if exp_cfg["intervention"] != "none" and exp_cfg["layer_to_experts"]:
            suppressor = ExpertSuppressor(
                model,
                layer_to_experts=exp_cfg["layer_to_experts"],
                factor=exp_cfg["factor"],
                mode=exp_cfg["intervention"],
            )
            suppressor.attach()

        recorder = MoERouteRecorder(model)
        recorder.attach()

        # 主循环
        todo = [(s, v) for s in samples for v in variants
                if (s["sample_id"], v.variant_id) not in done_set]
        log.info(f"本实验待处理: {len(todo)} 对")

        import torch
        n_done = 0
        n_failed = 0
        t_start = time.time()
        log_every = cfg["logging"].get("log_every_n_samples", 5)

        for i, (sample, variant) in enumerate(todo):
            try:
                result = run_one_sample(
                    model=model, processor=processor, config=model_config,
                    recorder=recorder,
                    sample=sample, variant=variant,
                    inference_cfg=inference_cfg,
                )
                # 在主记录上加 experiment 标签
                from dataclasses import replace
                # InferenceResult 是 dataclass，但 sample/variant 信息已经在里面
                # 我们改用 save 时直接 inject experiment_name
                _save_with_experiment(
                    result, exp_name,
                    main_records_path, routes_dir,
                    save_full_routes=False,   # Stage 2 主要看输出，不需要保存全部 routing
                )
                n_done += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_failed += 1
                log.exception(f"OOM: {sample['sample_id']}/{variant.variant_id}")
            except Exception as e:
                n_failed += 1
                log.exception(f"失败: {sample['sample_id']}/{variant.variant_id}: {e}")

            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
            if n_done % log_every == 0 and n_done > 0:
                rate = n_done / (time.time() - t_start)
                eta = (len(todo) - n_done) / rate if rate > 0 else float("inf")
                log.info(f"  [{exp_name}] 进度 {n_done}/{len(todo)} 失败 {n_failed} "
                         f"速率 {rate:.2f} 条/秒 ETA {eta/60:.1f} 分")

        recorder.detach()
        if suppressor:
            suppressor.detach()

        log.info(f"[{exp_name}] 完成: 成功 {n_done}, 失败 {n_failed}")
        log.info(f"  输出: {main_records_path}")

    log.info("=" * 70)
    log.info("所有实验完成！")
    log.info(f"用 scripts/09_stage2_analyze.py 对比 baseline vs 各实验")


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _load_done_set(main_records_path: str):
    done = set()
    p = Path(main_records_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["sample_id"], r["variant_id"]))
    return done


def _save_with_experiment(result, exp_name, main_records_path, routes_dir, save_full_routes):
    """写主记录 + 注入 experiment_name 字段；可选写 routes npz。"""
    from src.storage import _result_to_main_record

    record = _result_to_main_record(result)
    record["experiment_name"] = exp_name

    if save_full_routes and result.routes is not None:
        import numpy as np
        Path(routes_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(routes_dir) / result.route_file,
            expert_ids=result.routes["expert_ids"],
            gate_weights=result.routes["gate_weights"],
            phase_per_token=result.routes["phase_per_token"],
            modality_per_token=result.routes["modality_per_token"],
            chunk_lengths=np.array(result.routes["chunk_lengths"], dtype=np.int32),
        )
    else:
        # 不存 routes，把 route_file 字段清掉
        record["route_file"] = None

    Path(main_records_path).parent.mkdir(parents=True, exist_ok=True)
    with open(main_records_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""
正式 pilot 实验：对 pilot_index 里每条样本跑全部 prompt 变体并保存路由。

每条样本 × 7 个变体 (3 fast + 3 slow + 1 default)

跑法：
    cd /data/zhoukeru/q/TempoMoE
    python scripts/02_run_pilot.py

支持断点续跑：
    脚本会检查 main_records.jsonl 中已有的 (sample_id, variant_id) 对，
    跳过已完成的。

可选参数：
    --max_samples N      只跑前 N 条样本（覆盖全量）
    --datasets X,Y       只跑指定数据集
    --resume             显式启用断点续跑（默认开）
    --fresh              清空已有结果重跑
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Set, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--datasets", default=None, help="逗号分隔，如 RealWorldQA,MathVista")
    p.add_argument("--fresh", action="store_true", help="清空 runs/pilot 重跑")
    return p.parse_args()


def load_done_set(main_records_path: str) -> Set[Tuple[str, str]]:
    """读已完成的 (sample_id, variant_id) 集合。"""
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


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_pilot")

    # ---------- 读配置 ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 加载前先检查 GPU 空闲显存（排除别人占太多导致 OOM）
    from src.model_loader import (
        setup_visible_gpus, print_gpu_status, assert_enough_free_memory,
    )
    visible = cfg["gpu"]["visible_devices"]
    print_gpu_status(visible)
    assert_enough_free_memory(visible, cfg["gpu"].get("min_free_gib_per_gpu", 25))
    setup_visible_gpus(visible)

    # ---------- 准备路径 ----------
    main_records_path = cfg["paths"]["main_records"]
    routes_dir = cfg["paths"]["routes_dir"]
    run_dir = Path(cfg["paths"]["run_dir"])

    if args.fresh and run_dir.exists():
        log.warning(f"--fresh: 清空 {run_dir}")
        for f in run_dir.rglob("*"):
            if f.is_file():
                f.unlink()

    done_set = load_done_set(main_records_path)
    log.info(f"已完成 (sample, variant) 对: {len(done_set)} 条")

    # ---------- 加载样本索引 ----------
    from src.data_loader import load_pilot_index
    samples = load_pilot_index(cfg["paths"]["pilot_index"])
    log.info(f"pilot_index 总样本数: {len(samples)}")

    if args.datasets:
        keep = set(s.strip() for s in args.datasets.split(","))
        samples = [s for s in samples if s["dataset"] in keep]
        log.info(f"按数据集过滤后: {len(samples)} 条 (keep={keep})")

    if args.max_samples:
        samples = samples[: args.max_samples]
        log.info(f"按 max_samples 截取: {len(samples)} 条")

    # ---------- 加载 prompts ----------
    from src.prompts import load_prompts
    variants = load_prompts(cfg["paths"]["prompts_yaml"])
    log.info(f"prompt 变体数: {len(variants)}")

    # ---------- 加载模型 + hook ----------
    from src.model_loader import load_model
    from src.moe_hook import MoERouteRecorder

    model, processor, config = load_model(
        cfg["paths"]["model_dir"],
        cfg["gpu"]["max_memory_per_gpu"],
    )
    recorder = MoERouteRecorder(model, gate_class_name=cfg["hook"]["gate_class_name"])
    recorder.attach()

    # ---------- 主循环 ----------
    from src.inference import run_one_sample
    from src.storage import save_inference_result

    inference_cfg = dict(cfg["inference"])
    save_full_routes = cfg["hook"].get("save_full_routes", True)
    log_every = cfg["logging"].get("log_every_n_samples", 5)

    total_pairs = len(samples) * len(variants)
    todo = [
        (s, v) for s in samples for v in variants
        if (s["sample_id"], v.variant_id) not in done_set
    ]
    log.info(f"待处理: {len(todo)} / 总 {total_pairs}")

    import torch
    t_start = time.time()
    n_done = 0
    n_failed = 0
    n_oom_retried = 0

    def _try_run(sample, variant):
        """跑一条样本；OOM 时清 cache 重试一次。"""
        nonlocal n_oom_retried
        for attempt in range(2):
            try:
                return run_one_sample(
                    model=model, processor=processor, config=config,
                    recorder=recorder,
                    sample=sample, variant=variant,
                    inference_cfg=inference_cfg,
                )
            except torch.cuda.OutOfMemoryError:
                if attempt == 0:
                    n_oom_retried += 1
                    log.warning(
                        f"OOM 在 sample={sample['sample_id']} variant={variant.variant_id}，"
                        f"清 cache 后重试..."
                    )
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    continue
                raise

    for i, (sample, variant) in enumerate(todo):
        try:
            result = _try_run(sample, variant)
            save_inference_result(
                result=result,
                main_records_path=main_records_path,
                routes_dir=routes_dir,
                save_full_routes=save_full_routes,
            )
            n_done += 1
        except Exception as e:
            n_failed += 1
            log.exception(
                f"失败: sample={sample['sample_id']} variant={variant.variant_id}: {e}"
            )

        # 每 5 条样本主动清一次 cache，缓解碎片
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

        if n_done % log_every == 0 and n_done > 0:
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta = (len(todo) - n_done) / rate if rate > 0 else float("inf")
            # 顺便打印当前 GPU 占用，方便发现挤卡
            mem_lines = []
            for gid in range(torch.cuda.device_count()):
                allo = torch.cuda.memory_allocated(gid) / 1024**3
                rsvd = torch.cuda.memory_reserved(gid) / 1024**3
                mem_lines.append(f"cuda:{gid} alloc={allo:.1f}GiB rsvd={rsvd:.1f}GiB")
            log.info(
                f"进度 {n_done}/{len(todo)} 失败 {n_failed} OOM-重试 {n_oom_retried} "
                f"速率 {rate:.2f} 条/秒 ETA {eta/60:.1f} 分 | {' | '.join(mem_lines)}"
            )

    recorder.detach()

    log.info("=" * 50)
    log.info(f"pilot 完成: 成功 {n_done}, 失败 {n_failed}")
    log.info(f"主记录:   {main_records_path}")
    log.info(f"路由目录: {routes_dir}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()

"""
Smoke Test: 验证整个 pipeline 能跑通

只做一件事：
    用一张测试图 + 一个最简 prompt，跑一次推理，
    看模型加载、MoE hook、token 段定位、路由收集、写盘是否都正常。

成功标志（控制台会打印这些）：
    [OK] 模型加载
    [OK] 注册了 N 个 MoEGate hook  (Kimi-VL 应为 26)
    [OK] 推理产生了 X 个新 token
    [OK] 路由数据 shape = (26, T_total, 6)
    [OK] 路由 phase 分布: prompt=A, decode=B
    [OK] modality 分布: image=I, question=Q, instruction=N, decode=D
    [OK] 主记录已写入

跑法：
    cd /data/zhoukeru/q/TempoMoE
    python scripts/00_smoke_test.py

可选参数：
    --image  自定义测试图路径（默认用模型自带的 figures/demo.png）
    --gpus   覆盖配置里的可见 GPU
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# ===== 在 import torch 之前必须先设置 CUDA_VISIBLE_DEVICES =====
# 所以这里先解析参数，再做这一步，再 import torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--image", default=None, help="测试图路径，默认用模型自带 demo")
    p.add_argument("--question", default="Describe what you see in the image briefly.",
                   help="测试问题")
    p.add_argument("--gpus", default=None, help="覆盖 yaml 里的 GPU 设置，如 '6,7'")
    p.add_argument("--max_new_tokens", type=int, default=64,
                   help="smoke test 不需要长输出，64 足够")
    return p.parse_args()


def main():
    args = parse_args()

    # ---------- 读配置 ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    visible_devices = args.gpus or cfg["gpu"]["visible_devices"]

    # ---------- 配 logger ----------（先配 logger，下面打印才看得到）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("smoke_test")

    # ---------- 加载前先检查 GPU 空闲显存 ----------
    from src.model_loader import (
        setup_visible_gpus, print_gpu_status,
        assert_enough_free_memory, load_model,
    )
    print_gpu_status(visible_devices)
    min_free = cfg["gpu"].get("min_free_gib_per_gpu", 25)
    assert_enough_free_memory(visible_devices, min_free)

    # ---------- 设置 GPU 可见性（必须在 import torch 之前） ----------
    setup_visible_gpus(visible_devices)

    # ---------- 加载模型 ----------
    model, processor, config = load_model(
        cfg["paths"]["model_dir"],
        cfg["gpu"]["max_memory_per_gpu"],
    )
    print("[OK] 模型加载")

    # ---------- 注册 hook ----------
    from src.moe_hook import MoERouteRecorder
    recorder = MoERouteRecorder(model, gate_class_name=cfg["hook"]["gate_class_name"])
    n_hooks = recorder.attach()
    print(f"[OK] 注册了 {n_hooks} 个 MoEGate hook (期望 26)")

    # ---------- 准备一条 fake 样本 ----------
    if args.image is None:
        # 用模型自带的 demo 图
        candidate = PROJECT_ROOT / "model" / "model" / "Kimi-VL-A3B-Thinking-2506" / "figures"
        # 任取一张存在的图
        imgs = list(candidate.glob("*.png")) + list(candidate.glob("*.jpg"))
        if not imgs:
            raise FileNotFoundError(
                f"没找到 demo 图，请用 --image 指定一张测试图（{candidate} 下没有 png/jpg）"
            )
        image_path = str(imgs[0])
    else:
        image_path = args.image
    print(f"[INFO] 使用测试图: {image_path}")

    sample = {
        "sample_id": "smoke_0001",
        "dataset": "Smoke",
        "image_path": image_path,
        "question": args.question,
        "ground_truth": "",
    }

    # ---------- 准备一个最简 prompt 变体 ----------
    from src.prompts import PromptVariant
    variant = PromptVariant(
        prompt_type="default",
        variant_id="smoke_default",
        text="Answer the question based on the image.",
    )

    # ---------- 跑推理 ----------
    from src.inference import run_one_sample
    inference_cfg = dict(cfg["inference"])
    inference_cfg["max_new_tokens"] = args.max_new_tokens

    log.info("开始单样本推理 ...")
    result = run_one_sample(
        model=model,
        processor=processor,
        config=config,
        recorder=recorder,
        sample=sample,
        variant=variant,
        inference_cfg=inference_cfg,
    )
    print(f"[OK] 推理产生了 {result.output_token_count} 个新 token，"
          f"耗时 {result.total_latency:.2f}s")

    # ---------- 检查路由数据 ----------
    routes = result.routes
    expert_ids = routes["expert_ids"]
    print(f"[OK] 路由数据 shape = {expert_ids.shape}  (n_moe_layers, T_total, K)")
    print(f"     dtype: expert_ids={expert_ids.dtype}, "
          f"gate_weights={routes['gate_weights'].dtype}")

    phase = routes["phase_per_token"]
    n_prompt = int((phase == 0).sum())
    n_decode = int((phase == 1).sum())
    print(f"[OK] 路由 phase 分布: prompt={n_prompt}, decode={n_decode}")

    modality = routes["modality_per_token"]
    n_img = int((modality == 0).sum())
    n_q = int((modality == 1).sum())
    n_instr = int((modality == 2).sum())
    n_dec = int((modality == 3).sum())
    print(f"[OK] modality 分布: image={n_img}, question={n_q}, "
          f"instruction={n_instr}, decode={n_dec}")

    # 一个轻量的 sanity check：每 token 选 K=top_k 个不同专家
    K_expected = config.text_config.num_experts_per_tok
    K_actual = expert_ids.shape[2]
    assert K_actual == K_expected, f"top-k 不匹配: 期望 {K_expected}，实际 {K_actual}"

    # 看一下专家 id 范围
    n_exp = config.text_config.n_routed_experts
    assert expert_ids.min() >= 0 and expert_ids.max() < n_exp, \
        f"专家 id 越界: [{expert_ids.min()}, {expert_ids.max()}], 期望 [0, {n_exp})"
    print(f"[OK] 专家 id 范围合法: [{expert_ids.min()}, {expert_ids.max()}], "
          f"top-k={K_actual}")

    # ---------- 写盘到 runs/smoke ----------
    smoke_run_dir = PROJECT_ROOT / "runs" / "smoke"
    main_records_path = str(smoke_run_dir / "main_records.jsonl")
    routes_dir = str(smoke_run_dir / "routes")
    # 清掉历史 smoke 输出
    if smoke_run_dir.exists():
        for f in smoke_run_dir.rglob("*"):
            if f.is_file():
                f.unlink()

    from src.storage import save_inference_result
    save_inference_result(
        result=result,
        main_records_path=main_records_path,
        routes_dir=routes_dir,
        save_full_routes=True,
    )
    print(f"[OK] 主记录已写入: {main_records_path}")
    print(f"[OK] 路由 npz: {Path(routes_dir) / result.route_file}")

    # ---------- 显示模型输出片段 ----------
    print("\n----- 模型输出（前 300 字符）-----")
    print(result.model_output[:300])
    print("------------------------------------")
    print(f"\nparse_status: {result.parsed.parse_status}, "
          f"has_think_marker: {result.has_think_marker}")
    if result.has_think_marker:
        print(f"think_text 长度: {len(result.parsed.think_text)}")
        print(f"final_text: {result.parsed.final_text[:200]}")

    # ---------- 卸载 hook ----------
    recorder.detach()

    print("\n[Smoke Test 全部通过] ✓")


if __name__ == "__main__":
    main()

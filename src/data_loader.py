"""
数据加载模块：RealWorldQA + MathVista (testmini)

功能：
    1. 从 HuggingFace datasets 下载/缓存数据集
    2. 把图像保存到本地 data/pilot/images/，避免每次推理都从 HF 拉
    3. 输出统一的 pilot_index.jsonl，供 02_run_pilot.py 直接消费

pilot_index.jsonl 每行格式：
    {
      "sample_id":     "rwqa_0042",
      "dataset":       "RealWorldQA",
      "image_path":    "/abs/path/to/image.png",
      "question":      "...",
      "ground_truth":  "...",
      "extra":         {...}     # 数据集特有字段（题型、难度等）
    }
"""

import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# RealWorldQA
# ----------------------------------------------------------------------
def prepare_realworldqa(data_root: str, limit: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    准备 RealWorldQA 数据。

    huggingface 上的数据集名通常是 xai-org/RealworldQA。
    每条样本含：image (PIL.Image)、question、answer。

    Args:
        data_root: 项目数据根目录，会创建 data_root/pilot/images/realworldqa/
        limit: 采样数量
        seed: 随机种子

    Returns:
        list of dict，符合 pilot_index 格式
    """
    from datasets import load_dataset
    import random

    image_dir = Path(data_root) / "pilot" / "images" / "realworldqa"
    image_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 RealWorldQA ...")
    ds = load_dataset("xai-org/RealworldQA", split="test")

    # 全量打乱后取 limit 条
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    indices = indices[:limit]

    records = []
    for i, idx in enumerate(indices):
        item = ds[idx]
        image = item["image"]   # PIL.Image
        # 保存图像到本地
        img_path = image_dir / f"rwqa_{i:04d}.png"
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(img_path)

        records.append({
            "sample_id": f"rwqa_{i:04d}",
            "dataset": "RealWorldQA",
            "image_path": str(img_path.resolve()),
            "question": item["question"],
            "ground_truth": item.get("answer", ""),
            "extra": {"orig_idx": idx},
        })
    logger.info(f"RealWorldQA 准备完成: {len(records)} 条")
    return records


# ----------------------------------------------------------------------
# MathVista (testmini)
# ----------------------------------------------------------------------
def prepare_mathvista(data_root: str, limit: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    准备 MathVista testmini 数据。

    huggingface 上的数据集是 AI4Math/MathVista。
    我们用 testmini split（约 1000 条），有标准答案，适合 pilot。

    Args:
        data_root: 项目数据根目录
        limit: 采样数量
        seed: 随机种子
    """
    from datasets import load_dataset
    import random

    image_dir = Path(data_root) / "pilot" / "images" / "mathvista"
    image_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 MathVista (testmini) ...")
    ds = load_dataset("AI4Math/MathVista", split="testmini")

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    indices = indices[:limit]

    records = []
    for i, idx in enumerate(indices):
        item = ds[idx]
        image = item.get("decoded_image") or item.get("image")
        if image is None:
            logger.warning(f"MathVista idx={idx} 没有图像，跳过")
            continue
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_path = image_dir / f"mvista_{i:04d}.png"
        image.save(img_path)

        # MathVista 同时给了 query (含选项的完整问句) 和 question (原始问句)
        # 我们用 query，这样选择题选项也包含在内
        question = item.get("query") or item.get("question", "")

        records.append({
            "sample_id": f"mvista_{i:04d}",
            "dataset": "MathVista",
            "image_path": str(img_path.resolve()),
            "question": question,
            "ground_truth": str(item.get("answer", "")),
            "extra": {
                "orig_idx": idx,
                "question_type": item.get("question_type"),
                "answer_type": item.get("answer_type"),
                "choices": item.get("choices"),
            },
        })
    logger.info(f"MathVista 准备完成: {len(records)} 条")
    return records


# ----------------------------------------------------------------------
# 写入 pilot_index.jsonl
# ----------------------------------------------------------------------
def write_pilot_index(records: Iterable[Dict[str, Any]], out_path: str) -> int:
    """把记录写到 jsonl。返回写入条数。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    logger.info(f"已写入 {n} 条到 {out_path}")
    return n


def load_pilot_index(path: str) -> List[Dict[str, Any]]:
    """读取 pilot_index.jsonl。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

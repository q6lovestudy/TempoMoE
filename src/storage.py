"""
结果存储模块

写两类文件：
    1. main_records.jsonl —— 每行一条 (sample, variant) 主记录
    2. routes/{sample_id}__{variant_id}.npz —— 该样本该变体的完整路由数据

main_record 不含 routes 字段（已转存为 npz），通过 route_file 引用。
"""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .inference import InferenceResult

logger = logging.getLogger(__name__)


def save_inference_result(
    result: InferenceResult,
    main_records_path: str,
    routes_dir: str,
    save_full_routes: bool = True,
) -> None:
    """
    把一条 InferenceResult 落盘。

    - main 记录追加写到 main_records_path
    - 路由数据写到 routes_dir/{route_file}
    """
    routes_dir = Path(routes_dir)
    routes_dir.mkdir(parents=True, exist_ok=True)

    # 1) 写路由 npz
    if save_full_routes and result.routes is not None:
        npz_path = routes_dir / result.route_file
        np.savez_compressed(
            npz_path,
            expert_ids=result.routes["expert_ids"],
            gate_weights=result.routes["gate_weights"],
            phase_per_token=result.routes["phase_per_token"],
            modality_per_token=result.routes["modality_per_token"],
            chunk_lengths=np.array(result.routes["chunk_lengths"], dtype=np.int32),
        )

    # 2) 写主记录（不含 routes，但保留 route_file 引用）
    main_record = _result_to_main_record(result)
    Path(main_records_path).parent.mkdir(parents=True, exist_ok=True)
    with open(main_records_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(main_record, ensure_ascii=False) + "\n")


def _result_to_main_record(result: InferenceResult) -> Dict[str, Any]:
    """把 InferenceResult 序列化成 main_records.jsonl 一条记录的字典。"""
    return {
        "sample_id": result.sample_id,
        "dataset": result.dataset,
        "prompt_type": result.prompt_type,
        "variant_id": result.variant_id,
        "question": result.question,
        "image_path": result.image_path,
        "ground_truth": result.ground_truth,
        "prompt_text": result.prompt_text,
        "model_output": result.model_output,
        "parsed_answer": result.parsed.parsed_answer,
        "parse_status": result.parsed.parse_status,
        "has_think_marker": result.has_think_marker,
        "think_text": result.parsed.think_text,
        "final_text": result.parsed.final_text,
        "correct": None,  # 离线再算
        "input_token_count": result.input_token_count,
        "output_token_count": result.output_token_count,
        "token_segments": result.token_segments.to_dict(),
        "think_token_range": list(result.think_token_range) if result.think_token_range else None,
        "final_answer_token_range": list(result.final_answer_token_range) if result.final_answer_token_range else None,
        "decode_config": result.decode_config,
        "time_to_first_token": result.time_to_first_token,
        "total_latency": result.total_latency,
        "route_file": result.route_file,
    }


def load_main_records(path: str):
    """读 main_records.jsonl，生成器。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_routes(routes_dir: str, route_file: str) -> Dict[str, np.ndarray]:
    """读单条样本的 routes npz。"""
    p = Path(routes_dir) / route_file
    data = np.load(p, allow_pickle=False)
    return {k: data[k] for k in data.files}

"""
Kimi-VL-A3B-Thinking-2506 模型加载模块

职责：
    1. 通过环境变量限定可见 GPU
    2. 用 transformers 加载模型 + processor
    3. 暴露 (model, processor, config) 给上层调用

注意：
    - 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES，否则不生效
    - Kimi-VL 是自定义架构，必须 trust_remote_code=True
    - bf16 精度，~16B 参数双卡 22GiB 完全够
"""

import os
import logging
import subprocess
from typing import Tuple, Any, List, Optional

logger = logging.getLogger(__name__)


def setup_visible_gpus(visible_devices: str) -> None:
    """
    限定可见 GPU + 设置 PyTorch CUDA 分配器以减少碎片。
    必须在 import torch 之前调用。

    Args:
        visible_devices: 物理 GPU id 列表，逗号分隔，如 "4,5,6"
                         设置后程序内的 cuda:0 = 物理卡 4，依此类推
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    # expandable_segments 让分配器按需扩展段，减少长序列推理下的碎片化
    # PyTorch 自身在 OOM 报错里推荐这个设置
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logger.info(
        f"已设置 CUDA_VISIBLE_DEVICES={visible_devices}, "
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}"
    )


def query_gpu_free_memory_gib(physical_ids: Optional[List[int]] = None) -> List[Tuple[int, float, float]]:
    """
    通过 nvidia-smi 查询 GPU 空闲显存。

    必须在设置 CUDA_VISIBLE_DEVICES 之前用，因为 nvidia-smi 看的是物理 id。

    Args:
        physical_ids: 物理 GPU id 列表；None 表示全部 GPU
    Returns:
        list of (physical_id, free_gib, total_gib)
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().split("\n"):
        idx_s, free_s, total_s = [x.strip() for x in line.split(",")]
        idx, free_mib, total_mib = int(idx_s), float(free_s), float(total_s)
        if physical_ids is not None and idx not in physical_ids:
            continue
        rows.append((idx, free_mib / 1024.0, total_mib / 1024.0))
    return rows


def print_gpu_status(visible_devices: str) -> None:
    """加载模型前打印 visible_devices 里的卡的实时空闲显存，方便排错。"""
    ids = [int(x) for x in visible_devices.split(",") if x.strip()]
    info = query_gpu_free_memory_gib(ids)
    logger.info("可见 GPU 空闲显存（加载前）：")
    for idx, free, total in info:
        used = total - free
        logger.info(f"  物理 GPU {idx}: 已用 {used:5.1f} / 总 {total:5.1f} GiB / 空闲 {free:5.1f} GiB")


def assert_enough_free_memory(visible_devices: str, min_free_gib: float) -> None:
    """如果任何一张可见卡空闲 < min_free_gib，立刻报错并打印当前所有 GPU 状态以便换卡。"""
    ids = [int(x) for x in visible_devices.split(",") if x.strip()]
    info = query_gpu_free_memory_gib(ids)
    bad = [(idx, free) for idx, free, _ in info if free < min_free_gib]
    if not bad:
        return

    # 打印所有 GPU 状态供用户挑卡
    all_info = query_gpu_free_memory_gib()
    msg_lines = [
        f"可见 GPU 中有显存不足的卡（要求每张至少 {min_free_gib:.0f} GiB 空闲）：",
    ]
    for idx, free in bad:
        msg_lines.append(f"  物理 GPU {idx}: 空闲仅 {free:.1f} GiB")
    msg_lines.append("")
    msg_lines.append("当前所有 GPU 状态：")
    for idx, free, total in all_info:
        msg_lines.append(f"  GPU {idx}: 空闲 {free:5.1f} / 总 {total:5.1f} GiB")
    msg_lines.append("")
    msg_lines.append("挑两张最空的卡，改 configs/pilot.yaml 里的 visible_devices 后重跑。")
    raise RuntimeError("\n".join(msg_lines))


def load_model(model_dir: str, max_memory_per_gpu: str = "22GiB") -> Tuple[Any, Any, Any]:
    """
    加载 Kimi-VL 模型 + processor。

    Args:
        model_dir: 模型目录绝对路径
        max_memory_per_gpu: 每张可见卡上最多用的显存

    Returns:
        (model, processor, config)
        - model: KimiVLForConditionalGeneration，已 .eval()
        - processor: KimiVLProcessor，包含 tokenizer 和 image_processor
        - config: KimiVLConfig，方便上层读取层数、专家数等
    """
    # 在这里 import torch，确保 CUDA_VISIBLE_DEVICES 已生效
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

    n_visible = torch.cuda.device_count()
    logger.info(f"可见 GPU 数量: {n_visible}")
    if n_visible == 0:
        raise RuntimeError("未检测到可见 GPU，请先调用 setup_visible_gpus()")

    # 给 device_map=auto 提供每张卡的显存上限
    # cuda:0 和 cuda:1 是重映射后的逻辑 id
    max_memory = {i: max_memory_per_gpu for i in range(n_visible)}

    logger.info(f"开始加载模型: {model_dir}")
    logger.info(f"max_memory: {max_memory}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # transformers 4.50.3 对自定义 processor 不会自动从 chat_template.jinja 读模板
    # 这里手动兜底加载一下
    _ensure_chat_template(processor, model_dir)

    logger.info("模型加载完成")
    logger.info(
        f"语言模型层数: {config.text_config.num_hidden_layers}, "
        f"路由专家数: {config.text_config.n_routed_experts}, "
        f"共享专家数: {config.text_config.n_shared_experts}, "
        f"top-k: {config.text_config.num_experts_per_tok}, "
        f"first_k_dense_replace: {config.text_config.first_k_dense_replace}"
    )

    return model, processor, config


def _ensure_chat_template(processor, model_dir: str) -> None:
    """
    确保 processor 有 chat_template。
    Kimi-VL 的模板在 model_dir/chat_template.jinja，但部分版本 transformers 不会自动加载。
    """
    from pathlib import Path

    if getattr(processor, "chat_template", None):
        return  # 已加载，无需处理

    jinja_path = Path(model_dir) / "chat_template.jinja"
    if jinja_path.exists():
        processor.chat_template = jinja_path.read_text(encoding="utf-8")
        logger.info(f"从 {jinja_path.name} 手动加载 chat_template")
    else:
        logger.warning(
            f"未找到 chat_template，processor 也没自带；"
            f"调用 apply_chat_template 时会报错"
        )

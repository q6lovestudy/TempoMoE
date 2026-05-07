"""
Prompt 模板加载模块

从 configs/prompts.yaml 读取 fast/slow/default 三类 prompt 变体，
返回统一格式的 PromptVariant 列表。
"""

from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class PromptVariant:
    prompt_type: str    # "fast" / "slow" / "default"
    variant_id: str     # 如 "fast_v1"
    text: str           # 完整指令文本


def load_prompts(yaml_path: str) -> List[PromptVariant]:
    """
    读取 yaml 并展平成 PromptVariant 列表。

    返回顺序：fast_v1, fast_v2, fast_v3, slow_v1, ..., default_v1
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    variants: List[PromptVariant] = []
    for ptype in ["fast", "slow", "default"]:
        for item in data.get(ptype, []):
            variants.append(
                PromptVariant(
                    prompt_type=ptype,
                    variant_id=item["id"],
                    text=item["text"],
                )
            )
    return variants


def build_messages(instruction: str, question: str) -> list:
    """
    把指令文本和用户问题拼成 Kimi-VL chat template 期望的 messages 结构。

    指令放在用户消息的最前面（充当 system-style 引导），方便后续按段定位 token：
        [instruction] [question]

    image 由 processor 在 apply_chat_template 时通过 image=... 自动插入到 content 里。
    """
    user_content = [
        {"type": "image"},   # 占位，实际图像由 processor 填入
        {"type": "text", "text": f"{instruction}\n\n{question}"},
    ]
    return [{"role": "user", "content": user_content}]

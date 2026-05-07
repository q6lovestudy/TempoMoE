"""
模型输出解析模块

Kimi-VL-A3B-Thinking 的输出格式约定：
    <think_open> 思考过程 <think_close> 最终答案

注意：Kimi-VL-Thinking 永远会输出 think 标签，区别在于 think 段长度。
fast prompt 下 think 段会很短（例如 1-2 句话），不会消失。

异常情况：
    1. 模型忘记关闭 think 标签 -> unclosed_think
    2. think 标签存在但内容为空 -> empty_think（极少）
    3. 完全没有 think 标签 -> no_marker（极少；模型一般都会标）
    4. 多个 think 块嵌套 -> 取第一个

解析结果会附在主记录里，方便后期过滤和分组。
"""

import re
from dataclasses import dataclass, asdict
from typing import Optional


# 这些是 Kimi-VL tokenizer 的特殊 token 字符串形式（decode 时 skip_special_tokens=False
# 保留下来的）。final_text / parsed_answer 不应包含这些。
_SPECIAL_TOKEN_PATTERNS = [
    re.compile(r"<\|im_end\|>"),
    re.compile(r"<\|im_user\|>"),
    re.compile(r"<\|im_assistant\|>"),
    re.compile(r"<\|im_system\|>"),
    re.compile(r"<\|im_middle\|>"),
    re.compile(r"<\|media_start\|>"),
    re.compile(r"<\|media_end\|>"),
    re.compile(r"<\|media_content\|>"),
    re.compile(r"<\|media_pad\|>"),
    re.compile(r"\[EOS\]"),
    re.compile(r"\[BOS\]"),
    re.compile(r"\[PAD\]"),
]


def _strip_special_tokens(text: str) -> str:
    """剥掉模型输出末尾常见的特殊 token 字符串残留。"""
    for pat in _SPECIAL_TOKEN_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


@dataclass
class ParsedOutput:
    parse_status: str          # "ok" / "unclosed_think" / "empty_think" / "no_marker"
    has_think_marker: bool
    think_text: str            # 思考段（去掉标签）
    final_text: str            # 最终回答段
    parsed_answer: str         # 从 final_text 中尝试抽取的简洁答案

    def to_dict(self) -> dict:
        return asdict(self)


def parse_output(
    raw: str,
    think_open: str = "◁think▷",
    think_close: str = "◁/think▷",
) -> ParsedOutput:
    """
    解析模型原始输出。

    Args:
        raw: 模型 generate 出来的字符串（不含输入 prompt）
        think_open / think_close: thinking 标签

    Returns:
        ParsedOutput
    """
    # 没有 open 标签
    if think_open not in raw:
        clean = _strip_special_tokens(raw)
        return ParsedOutput(
            parse_status="no_marker",
            has_think_marker=False,
            think_text="",
            final_text=clean,
            parsed_answer=_extract_concise_answer(clean),
        )

    # 有 open 标签，找 close
    parts = raw.split(think_open, 1)
    pre_think = parts[0]
    rest = parts[1]

    if think_close not in rest:
        return ParsedOutput(
            parse_status="unclosed_think",
            has_think_marker=True,
            think_text=_strip_special_tokens(rest),
            final_text=_strip_special_tokens(pre_think),
            parsed_answer=_extract_concise_answer(_strip_special_tokens(pre_think)),
        )

    think_part, final_part = rest.split(think_close, 1)
    think_text = _strip_special_tokens(think_part)
    final_text = _strip_special_tokens(pre_think + final_part)

    if not think_text:
        return ParsedOutput(
            parse_status="empty_think",
            has_think_marker=True,
            think_text="",
            final_text=final_text,
            parsed_answer=_extract_concise_answer(final_text),
        )

    return ParsedOutput(
        parse_status="ok",
        has_think_marker=True,
        think_text=think_text,
        final_text=final_text,
        parsed_answer=_extract_concise_answer(final_text),
    )


# ----------------------------------------------------------------------
# 简洁答案抽取（启发式，不强求精确）
# ----------------------------------------------------------------------
_ANSWER_PATTERNS = [
    # "The answer is X" / "Answer: X" 等
    re.compile(r"(?:final answer|the answer is|answer is|answer:)\s*[:\-]?\s*(.+?)(?:\.|$)", re.IGNORECASE | re.DOTALL),
    # **X** 形式
    re.compile(r"\*\*(.+?)\*\*"),
    # \boxed{X} LaTeX 形式（数学题常见）
    re.compile(r"\\boxed\{(.+?)\}"),
]


def _extract_concise_answer(text: str) -> str:
    """启发式抽取简洁答案；抽不出来就返回 final_text 截断版本。"""
    text = text.strip()
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip().strip(".,")
    # 兜底：取前 200 字符
    return text[:200]

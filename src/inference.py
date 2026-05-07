"""
单样本推理 + 路由记录模块

主流程：
    1. 把 (image, instruction, question) 喂给 processor
    2. 在 generate() 前 hook MoEGate；用 record 记录每次 forward 的 routing
    3. generate() 用 greedy；把输出文本和 routing 合并成完整记录
    4. 计算 token 段边界（image / question / instruction / decode）

token 段定位策略：
    - 我们利用 image_processor 处理后的 input_ids 直接定位 image token
      （Kimi-VL 的 image token id = config.media_placeholder_token_id = 163605）
    - 文本部分通过分别 tokenize "instruction" 和 "question" 来计算长度
      然后按已知顺序拼成 [chat_template_prefix] [image] [instruction] [question] [chat_template_suffix]
    - 由于 chat template 在两端加了固定 token，我们用模板 token 总长度反推位置

更稳妥的实现：直接对最终 prompt token 做扫描定位 image token，剩余部分按段长度推断。
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .moe_hook import MoERouteRecorder
from .parser import parse_output, ParsedOutput
from .prompts import PromptVariant, build_messages

logger = logging.getLogger(__name__)


@dataclass
class TokenSegments:
    """每段 token 的 [start, end) 区间（在完整序列中的位置）"""
    image:       Tuple[int, int]
    question:    Tuple[int, int]
    instruction: Tuple[int, int]
    decode:      Tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "image":       list(self.image),
            "question":    list(self.question),
            "instruction": list(self.instruction),
            "decode":      list(self.decode),
        }

    def modality_per_token(self, T_total: int) -> np.ndarray:
        """生成 [T_total] 的 modality 标记:
           0=image 1=question 2=instruction 3=decode 4=other(系统前缀/boundary)
           默认 4，避免没标到的 token 被错算成 image。
           越界部分（segments end > T_total）会被 numpy 切片自动忽略。
        """
        m = np.full(T_total, 4, dtype=np.int8)
        # 用 min(end, T_total) 防止越界
        def _set(start, end, val):
            s, e = max(0, start), min(end, T_total)
            if s < e:
                m[s:e] = val
        _set(self.image[0], self.image[1], 0)
        _set(self.question[0], self.question[1], 1)
        _set(self.instruction[0], self.instruction[1], 2)
        _set(self.decode[0], self.decode[1], 3)
        return m


@dataclass
class InferenceResult:
    sample_id: str
    dataset: str
    prompt_type: str
    variant_id: str
    question: str
    image_path: str
    ground_truth: str
    prompt_text: str               # 完整指令文本
    model_output: str              # 模型生成的全部内容（不含 prompt）
    parsed: ParsedOutput
    input_token_count: int
    output_token_count: int
    token_segments: TokenSegments
    has_think_marker: bool
    think_token_range: Optional[Tuple[int, int]]
    final_answer_token_range: Optional[Tuple[int, int]]
    decode_config: Dict[str, Any]
    time_to_first_token: float
    total_latency: float
    # 路由数据另存为 npz，这里只放文件名
    route_file: str
    # 完整路由数据（写盘前临时持有，写完置 None）
    routes: Optional[Dict[str, np.ndarray]] = None


# ----------------------------------------------------------------------
# token 段定位
# ----------------------------------------------------------------------
def locate_token_segments(
    input_ids: torch.Tensor,
    instruction: str,
    question: str,
    processor,
    media_placeholder_token_id: int,
) -> TokenSegments:
    """
    在完整 input_ids（prompt 部分）里定位四段 token 的边界。
    decode 段的 end 由调用方在 generate 后填。

    定位策略（用特殊 token 做锚点，比子序列匹配可靠）：
        chat template 渲染后的 prompt 结构：
            <|im_system|>...<|im_end|>          ← 系统前缀
            <|im_user|>user<|im_middle|>
            <|media_start|>image<|media_content|>
                <|media_pad|>×N                  ← 图像 token (N 由图像决定)
            <|media_end|>
            {instruction}\n\n{question}          ← 文本段，紧跟在 <|media_end|> 后
            <|im_end|>
            <|im_assistant|>assistant<|im_middle|>   ← 生成提示后缀

        所以：
        - image 段 = 连续的 <|media_pad|> token（id = media_placeholder_token_id）
        - 文本段起点 = image 段结束之后的下一个 token（紧跟 <|media_end|>）
        - 文本段终点 = 文本段起点之后第一个 <|im_end|>
        - instruction 与 question 之间用 token 长度切分

    Returns:
        TokenSegments，其中 decode 段先填 (T_input, T_input)
    """
    ids = input_ids.tolist()
    T_input = len(ids)
    tok = processor.tokenizer

    # 1) image 段：连续 media_placeholder
    img_start = img_end = -1
    for i, t in enumerate(ids):
        if t == media_placeholder_token_id:
            if img_start == -1:
                img_start = i
            img_end = i + 1
        elif img_start != -1:
            break
    if img_start == -1:
        logger.warning("未找到 image token，输入格式异常")
        img_start = img_end = 0

    # 2) 找文本段的右边界：紧跟 image 段之后第一个 <|im_end|>
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id < 0:
        # 兜底：扫到序列末尾
        text_segment_end = T_input
    else:
        text_segment_end = T_input
        for i in range(img_end, T_input):
            if ids[i] == im_end_id:
                text_segment_end = i
                break

    # 文本段在 image 段结束 与 第一个 <|im_end|> 之间，注意 image 段后通常还有
    # 1 个 <|media_end|> token，文本从它之后开始。但 <|media_end|> 不是 media_pad，
    # 所以 image 段在 media_pad 结束就停了，text_start 就是 img_end + 0 或 +1。
    # 简化：直接从 img_end 开始找；前面少量 boundary token 默认归入 modality=4。

    # 3) 在文本段内切分 instruction / question
    # 用 token 长度切分（相对位置可能有 ±1 的 BPE 边界误差，但不影响主体分布）
    instr_ids = tok(instruction, add_special_tokens=False)["input_ids"]
    q_ids = tok(question, add_special_tokens=False)["input_ids"]

    # instruction 紧贴文本段起点（用 img_end 作为粗起点；boundary token 进 modality=4）
    instr_start = img_end
    instr_end = min(instr_start + len(instr_ids), text_segment_end)

    # question 紧贴 instruction 之后（中间 \n\n 默认归入 modality=4）
    q_start = max(instr_end, text_segment_end - len(q_ids))
    q_end = min(q_start + len(q_ids), text_segment_end)

    return TokenSegments(
        image=(img_start, img_end),
        question=(q_start, q_end),
        instruction=(instr_start, instr_end),
        decode=(T_input, T_input),
    )


def _find_subsequence(haystack: List[int], needle: List[int], start: int = 0) -> int:
    """在 haystack[start:] 里查找 needle 的起点，找不到返回 -1。"""
    if not needle:
        return -1
    n_h, n_n = len(haystack), len(needle)
    for i in range(start, n_h - n_n + 1):
        if haystack[i:i + n_n] == needle:
            return i
    return -1


# ----------------------------------------------------------------------
# 单样本推理
# ----------------------------------------------------------------------
def run_one_sample(
    model,
    processor,
    config,
    recorder: MoERouteRecorder,
    sample: Dict[str, Any],
    variant: PromptVariant,
    inference_cfg: Dict[str, Any],
) -> InferenceResult:
    """
    对一条样本 + 一个 prompt 变体跑一次推理，返回完整记录（含路由）。

    Args:
        model, processor, config: 由 model_loader 提供
        recorder: 已 attach 的 MoERouteRecorder（调用前不需要 reset，本函数会 reset）
        sample: pilot_index 一条
        variant: PromptVariant
        inference_cfg: 推理配置 dict（do_sample, max_new_tokens 等）
    """
    # ---------- 1. 准备输入 ----------
    image = Image.open(sample["image_path"])
    if image.mode != "RGB":
        image = image.convert("RGB")

    messages = build_messages(variant.text, sample["question"])

    # apply_chat_template 渲染成纯文本（不要传 return_tensors，下面 processor() 才负责 tokenize）
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)

    # 把所有张量移到模型主设备（device_map=auto 时是 cuda:0）
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    input_ids = inputs["input_ids"][0]   # shape [seq_len]
    T_input = input_ids.shape[0]

    # ---------- 2. token 段定位 ----------
    segments = locate_token_segments(
        input_ids,
        instruction=variant.text,
        question=sample["question"],
        processor=processor,
        media_placeholder_token_id=config.media_placeholder_token_id,
    )

    # ---------- 3. 推理 ----------
    recorder.reset()

    gen_kwargs = dict(
        do_sample=inference_cfg.get("do_sample", False),
        max_new_tokens=inference_cfg.get("max_new_tokens", 1024),
        temperature=inference_cfg.get("temperature", 1.0) if inference_cfg.get("do_sample") else None,
        top_p=inference_cfg.get("top_p", 1.0),
        return_dict_in_generate=True,
        output_scores=False,
    )
    # greedy 时 transformers 不允许传 temperature
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    total_latency = time.time() - t0

    seqs = out.sequences if hasattr(out, "sequences") else out
    output_ids = seqs[0][T_input:]   # 去掉 prompt 部分
    T_output = output_ids.shape[0]
    model_output = processor.tokenizer.decode(output_ids, skip_special_tokens=False)

    # 修正 decode 段
    segments.decode = (T_input, T_input + T_output)

    # ---------- 4. 收集路由 ----------
    routes = recorder.collect()
    T_total = routes["expert_ids"].shape[1]

    # KV-cache 推理的正常情况：最后生成的 token 不会再走 forward，
    # 所以 hook 看到的总 token = T_input + T_output - 1（早停时还可能再少 1）。
    # 只在差异大于 2 时才警告，否则属于预期行为。
    diff = (T_input + T_output) - T_total
    if diff < 0 or diff > 2:
        logger.warning(
            f"路由 token 总数 ({T_total}) 与 input+output ({T_input + T_output}) "
            f"差异异常（diff={diff}），后续按路由数据为准。"
        )

    # 按真实 T_total 重新生成 modality 标记，并把 decode 段对齐到实际长度
    actual_decode_end = T_total
    segments.decode = (T_input, actual_decode_end)
    routes["modality_per_token"] = segments.modality_per_token(T_total)

    # ---------- 5. 解析输出 ----------
    parsed = parse_output(
        model_output,
        think_open=inference_cfg.get("think_open_token", "◁think▷"),
        think_close=inference_cfg.get("think_close_token", "◁/think▷"),
    )

    # think / final 在 token 层的范围（文本层粗定位 -> token 索引）
    think_range, final_range = _locate_think_final_in_decode(
        model_output, parsed, processor.tokenizer, T_input
    )

    # ---------- 6. 构造返回 ----------
    route_file = f"{sample['sample_id']}__{variant.variant_id}.npz"
    return InferenceResult(
        sample_id=sample["sample_id"],
        dataset=sample["dataset"],
        prompt_type=variant.prompt_type,
        variant_id=variant.variant_id,
        question=sample["question"],
        image_path=sample["image_path"],
        ground_truth=sample.get("ground_truth", ""),
        prompt_text=variant.text,
        model_output=model_output,
        parsed=parsed,
        input_token_count=T_input,
        output_token_count=T_output,
        token_segments=segments,
        has_think_marker=parsed.has_think_marker,
        think_token_range=think_range,
        final_answer_token_range=final_range,
        decode_config={
            "do_sample": inference_cfg.get("do_sample", False),
            "temperature": inference_cfg.get("temperature", 0.0),
            "top_p": inference_cfg.get("top_p", 1.0),
            "max_new_tokens": inference_cfg.get("max_new_tokens", 1024),
        },
        time_to_first_token=-1.0,   # 第一阶段不细测，留 -1
        total_latency=total_latency,
        route_file=route_file,
        routes=routes,
    )


def _locate_think_final_in_decode(
    model_output: str,
    parsed: ParsedOutput,
    tokenizer,
    T_input: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    通过 tokenize 子串大致定位 think 段和 final 段在完整序列里的 token 范围。
    粗定位即可，离线分析时用 modality 标记 + token 内容也能复核。
    """
    if not parsed.has_think_marker:
        # 全部是 final
        out_ids = tokenizer(model_output, add_special_tokens=False)["input_ids"]
        return None, (T_input, T_input + len(out_ids))

    # 用文本切分定位
    think_ids = tokenizer(parsed.think_text, add_special_tokens=False)["input_ids"] if parsed.think_text else []
    final_ids = tokenizer(parsed.final_text, add_special_tokens=False)["input_ids"] if parsed.final_text else []

    # 粗略：think 段紧接在 input 之后（前面可能有 ◁think▷ 标签 token，忽略）
    think_start = T_input
    think_end = think_start + len(think_ids)
    final_start = think_end
    final_end = final_start + len(final_ids)
    return (think_start, think_end), (final_start, final_end)

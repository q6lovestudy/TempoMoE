"""
MoE 路由 hook 模块

核心思路：
    1. Kimi-VL 每层 MoE 的入口是 DeepseekV3MoE.gate (类名 MoEGate)
    2. MoEGate.forward(hidden_states) 返回 (topk_idx, topk_weight, aux_loss)
        - topk_idx:    [bsz*seq_len, K=6] 选中的专家 id
        - topk_weight: [bsz*seq_len, K=6] 归一化后的门权重 (× routed_scaling_factor)
    3. 在每个 MoEGate 上注册 forward_hook，截取这两个张量

phase 判定：
    - 第一次 forward = prefill，seq_len = 输入 token 数
    - 后续每次 forward = decode 一步，seq_len = 1
    - 我们用 chunk 长度区分，第一个 chunk 标 phase=prompt，其余标 phase=decode

使用流程：
    recorder = MoERouteRecorder(model, n_routed_experts=64, top_k=6)
    recorder.attach()
    recorder.reset()                # 每条样本前清空
    out = model.generate(...)       # 推理
    routes = recorder.collect()     # 拿到聚合后的路由数据
    recorder.detach()               # 实验结束统一卸载

注意：
    - shared_experts 不走 gate，hook 不会捕获到它们 -> 自然排除
    - 只有 layer_idx >= first_k_dense_replace 的层才有 MoE，dense 层不会触发 hook
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GateCapture:
    """单个 MoE 层在一次推理中的全部捕获结果（多次 forward 拼接前的原始 chunk 列表）"""
    expert_ids_chunks: List[np.ndarray] = field(default_factory=list)   # 每元素 shape [T_chunk, K]
    gate_weights_chunks: List[np.ndarray] = field(default_factory=list) # 同上
    chunk_lengths: List[int] = field(default_factory=list)              # 每次 forward 的 token 数


class MoERouteRecorder:
    """
    给所有 MoEGate 注册 forward hook，按层、按 forward 调用累积捕获结果。
    """

    def __init__(self, model: torch.nn.Module, gate_class_name: str = "MoEGate"):
        """
        Args:
            model: 加载好的 Kimi-VL 模型
            gate_class_name: 门控模块的类名，用于按类名检索
        """
        self.model = model
        self.gate_class_name = gate_class_name

        # layer_name -> GateCapture
        self.captures: Dict[str, GateCapture] = {}
        # layer_name -> hook handle，用于卸载
        self._hook_handles = []
        # layer_name -> layer_idx（按发现顺序赋 0..N-1）
        self.layer_name_to_idx: Dict[str, int] = {}
        # 反向索引方便外部查询
        self.layer_idx_to_name: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # 安装 / 卸载
    # ------------------------------------------------------------------
    def attach(self) -> int:
        """
        遍历模型，找到所有 MoEGate 子模块并注册 forward hook。

        Returns:
            注册的 hook 数量（= MoE 层数，Kimi-VL 应为 26）
        """
        if self._hook_handles:
            raise RuntimeError("已有 hook 注册，请先调用 detach()")

        gate_modules = []
        for name, module in self.model.named_modules():
            # 用类名匹配，避免依赖 import 路径
            if module.__class__.__name__ == self.gate_class_name:
                gate_modules.append((name, module))

        if not gate_modules:
            raise RuntimeError(
                f"在模型里找不到类名为 {self.gate_class_name} 的模块，"
                f"请检查 modeling 文件"
            )

        # 按发现顺序赋层索引（named_modules 按前序遍历，正好是层序）
        for layer_idx, (name, module) in enumerate(gate_modules):
            self.layer_name_to_idx[name] = layer_idx
            self.layer_idx_to_name[layer_idx] = name
            self.captures[name] = GateCapture()
            handle = module.register_forward_hook(self._make_hook(name))
            self._hook_handles.append(handle)

        logger.info(f"已注册 {len(self._hook_handles)} 个 MoEGate hook")
        return len(self._hook_handles)

    def detach(self) -> None:
        """卸载所有 hook"""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        logger.info("所有 hook 已卸载")

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """每条样本推理前清空累积状态"""
        for cap in self.captures.values():
            cap.expert_ids_chunks.clear()
            cap.gate_weights_chunks.clear()
            cap.chunk_lengths.clear()

    # ------------------------------------------------------------------
    # hook 工厂
    # ------------------------------------------------------------------
    def _make_hook(self, layer_name: str):
        """
        生成一个闭包 hook，捕获该层的 (topk_idx, topk_weight)。

        MoEGate.forward 返回 tuple(topk_idx, topk_weight, aux_loss)
        torch hook 收到的 output 就是这个 tuple
        """
        cap = self.captures[layer_name]

        def hook(module, inputs, output):
            # output: (topk_idx, topk_weight, aux_loss)
            topk_idx, topk_weight, _ = output

            # shape: [bsz*seq_len, K]，bsz=1 推理时 = [seq_len, K]
            ids = topk_idx.detach().to("cpu", dtype=torch.int16).numpy()
            wts = topk_weight.detach().to("cpu", dtype=torch.float16).numpy()

            cap.expert_ids_chunks.append(ids)
            cap.gate_weights_chunks.append(wts)
            cap.chunk_lengths.append(ids.shape[0])

        return hook

    # ------------------------------------------------------------------
    # 收集结果
    # ------------------------------------------------------------------
    def collect(self) -> Dict[str, np.ndarray]:
        """
        把当前样本的所有捕获 chunk 拼成完整张量。

        Returns:
            dict 包含：
                expert_ids:    int16  [n_moe_layers, T_total, K]
                gate_weights:  float16 [n_moe_layers, T_total, K]
                phase_per_token: int8 [T_total]    0=prompt, 1=decode
                layer_names: list[str]              # 按 layer_idx 排序
                chunk_lengths: list[int]            # 第一个 chunk 是 prefill 长度
        """
        if not self.captures:
            raise RuntimeError("还没 attach 或没有触发任何 hook")

        # 任取一层拿 chunk 长度（所有 MoE 层的 chunk 长度应相同）
        any_layer = next(iter(self.captures.values()))
        chunk_lengths = list(any_layer.chunk_lengths)

        if not chunk_lengths:
            raise RuntimeError("没有捕获到任何 routing；模型是否前向了一次？")

        # 一致性检查：每层 chunk 长度序列应该一致
        for name, cap in self.captures.items():
            if cap.chunk_lengths != chunk_lengths:
                raise RuntimeError(
                    f"层 {name} 的 chunk 长度序列与其它层不一致："
                    f"{cap.chunk_lengths} vs {chunk_lengths}"
                )

        # 第一个 chunk = prefill (prompt phase)，后续每个 chunk = decode 一步
        T_total = sum(chunk_lengths)
        phase_per_token = np.zeros(T_total, dtype=np.int8)
        T_prompt = chunk_lengths[0]
        phase_per_token[T_prompt:] = 1  # decode phase

        # 按 layer_idx 顺序排列
        n_layers = len(self.captures)
        layer_names = [self.layer_idx_to_name[i] for i in range(n_layers)]

        # 拼接：每层得到 [T_total, K]
        K = any_layer.expert_ids_chunks[0].shape[1]
        expert_ids = np.zeros((n_layers, T_total, K), dtype=np.int16)
        gate_weights = np.zeros((n_layers, T_total, K), dtype=np.float16)

        for layer_idx, name in enumerate(layer_names):
            cap = self.captures[name]
            ids_concat = np.concatenate(cap.expert_ids_chunks, axis=0)
            wts_concat = np.concatenate(cap.gate_weights_chunks, axis=0)
            expert_ids[layer_idx] = ids_concat
            gate_weights[layer_idx] = wts_concat

        return {
            "expert_ids": expert_ids,
            "gate_weights": gate_weights,
            "phase_per_token": phase_per_token,
            "layer_names": layer_names,
            "chunk_lengths": chunk_lengths,
        }

from __future__ import annotations
from typing import Sequence, Any
import torch
from .llama_proxy import LlamaProxyModel

class LlamaSharedProxyModel(LlamaProxyModel):
    """
    一个支持共享特征提取的 Llama Proxy Model 扩展类。
    
    它继承自 LlamaProxyModel，并公开了特征提取接口，
    允许外部选择器（SharedLlamaSelector）复用其底层 Transformer 计算结果。
    """
    def extract_features_tensor(self, inputs: Sequence[Any]) -> torch.Tensor:
        """
        公开的特征提取接口。
        
        Args:
            inputs: 文本输入列表
            
        Returns:
            torch.Tensor: Shape (B, HiddenSize)，在当前 device 上。
        """
        # 强制不计算梯度，因为这只是特征提取
        with torch.no_grad():
            features = self._extract_features(inputs)
            # 确保切断任何可能的历史（虽然 no_grad 应该够了，但双重保险）
            return features.detach()

    @property
    def hidden_dim(self) -> int:
        """返回 Hidden Layer 维度，方便 Selector 初始化 Head"""
        if hasattr(self, "hidden_size") and int(getattr(self, "hidden_size", 0)) > 0:
            return int(self.hidden_size)
        # Fallback logic if hidden_size was not set or is invalid
        return int(getattr(self.model.config, "hidden_size", 0))

"""SharedLlamaSelectorV2：在 V1 基础上增加 Replay Buffer，解决每轮训练数据稀少问题。

主要改进：
- Replay Buffer：保留最近 `buffer_maxlen` 条 (feature, label) 对，每轮训练时
  将新提取的特征与 buffer 中的历史特征合并后再训练 Head，稳定梯度信号。
- 其余接口与 SharedLlamaSelector 完全兼容，可直接替换。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseSelector


class _ClassificationHead(nn.Module):
    """两层 MLP 分类头"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SharedLlamaSelectorV2(BaseSelector):
    """Replay Buffer 版本的 Shared Llama Selector。

    与 V1 的区别：
    1. 内部维护一个 Replay Buffer（deque），保留最近 `buffer_maxlen` 条
       (feature_vec, pseudo_label) 对。
    2. 每次 update() 时，先提取当前 batch 的特征，加入 buffer，
       然后用 buffer 全量数据重训 head（而不仅仅用当前 batch 的 10 条）。
    3. `buffer_maxlen=0` 时退化为与 V1 完全相同的行为（不开 buffer）。
    """

    def __init__(
        self,
        proxy_model: Any,          # LlamaSharedProxyModel
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 16,
        inference_batch_size: int = 4,
        buffer_maxlen: int = 1000,  # 0 = 关闭 replay buffer
    ) -> None:
        self.proxy = proxy_model
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.buffer_maxlen = int(buffer_maxlen)
        self._use_replay_buffer = self.buffer_maxlen > 0

        # 推断 hidden_size
        if hasattr(self.proxy, "hidden_dim"):
            self.input_dim = self.proxy.hidden_dim
        elif hasattr(self.proxy, "hidden_size"):
            self.input_dim = self.proxy.hidden_size
        else:
            self.input_dim = 4096
            if hasattr(self.proxy, "model") and hasattr(self.proxy.model, "config"):
                self.input_dim = getattr(self.proxy.model.config, "hidden_size", 4096)

        self.device = self.proxy.device

        self.head = _ClassificationHead(
            self.input_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        ).to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Replay Buffer: 存储 CPU float32 张量 (1, H) 和 float 标签
        # 注意：buffer_maxlen=0 表示关闭 replay buffer（退化为“仅用当前 batch 训练”）。
        # 不能用 deque(maxlen=None) 来表达“关闭”，那会导致无限增长。
        self._buf_features: Deque[torch.Tensor] = deque(maxlen=self.buffer_maxlen) if self._use_replay_buffer else deque(maxlen=0)
        self._buf_labels: Deque[float] = deque(maxlen=self.buffer_maxlen) if self._use_replay_buffer else deque(maxlen=0)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def score(self, inputs: Sequence[Any]) -> np.ndarray:
        """返回 (0,1) 分数，越高越值得查询。"""
        self.head.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(inputs), self.inference_batch_size):
                batch = inputs[i : i + self.inference_batch_size]
                feats = self.proxy.extract_features_tensor(batch).detach()
                logits = self.head(feats.float())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)

    def update(
        self,
        inputs: Sequence[Any],
        labels: np.ndarray,
        epochs: int = 1,
        batch_size: int = 0,
        **kwargs: Any,
    ) -> None:
        """提取特征 → 加入 Buffer → 用 Buffer 全量重训 Head。"""
        if batch_size <= 0:
            batch_size = self.batch_size

        # 1. 提取当前 batch 特征
        new_features: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(inputs), self.inference_batch_size):
                batch = inputs[i : i + self.inference_batch_size]
                feats = self.proxy.extract_features_tensor(batch).detach().float().cpu()
                for j in range(feats.shape[0]):
                    new_features.append(feats[j : j + 1])  # shape (1, H)

        if not new_features:
            return

        if self._use_replay_buffer:
            # 2. 写入 Replay Buffer
            for feat, lbl in zip(new_features, labels.tolist()):
                self._buf_features.append(feat)
                self._buf_labels.append(float(lbl))

            # 3. 用 Buffer 全量构建训练集
            all_feats = torch.cat(list(self._buf_features), dim=0)   # (N, H)
            all_lbls = torch.tensor(list(self._buf_labels), dtype=torch.float32)  # (N,)
        else:
            # buffer 关闭：退化为仅使用当前 batch 的特征进行训练
            all_feats = torch.cat(new_features, dim=0)  # (N, H)
            all_lbls = torch.tensor(labels, dtype=torch.float32)  # (N,)

        dataset = TensorDataset(all_feats, all_lbls)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 4. 训练 Head
        self.head.train()
        for _ in range(epochs):
            for batch_feats, batch_lbls in loader:
                batch_feats = batch_feats.to(self.device)
                batch_lbls = batch_lbls.to(self.device)
                self.optimizer.zero_grad()
                logits = self.head(batch_feats)
                loss = self.loss_fn(logits, batch_lbls)
                loss.backward()
                self.optimizer.step()

    def reset_head(self) -> None:
        """Cold-restart head 权重（用于应对特征漂移）。"""
        for m in self.head.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        # 重建 optimizer（绑定到新参数状态）
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.optimizer.param_groups[0]["lr"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
        )

    def buffer_size(self) -> int:
        return len(self._buf_features) if self._use_replay_buffer else 0

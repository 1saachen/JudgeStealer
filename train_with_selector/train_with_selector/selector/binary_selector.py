from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
	from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
	AutoModel = None  # type: ignore[assignment]
	AutoTokenizer = None  # type: ignore[assignment]

from .base import BaseSelector


class _ClassificationHead(nn.Module):
	"""分类头结构 (Classification Head).

	接收特征向量（如 BERT 隐藏层输出），输出 Logits。
	升级为更深层的 MLP 结构：Linear -> LayerNorm -> ReLU -> Dropout ...
	"""

	def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
		super().__init__()
		self.net = nn.Sequential(
			# 第一层：特征变换与映射 (in -> hidden*2)
			nn.Linear(in_dim, hidden_dim * 2),
			nn.LayerNorm(hidden_dim * 2),
			nn.ReLU(),
			nn.Dropout(dropout),

			# 第二层：特征压缩与抽象 (hidden*2 -> hidden)
			nn.Linear(hidden_dim * 2, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),

			# 第三层：输出层
			nn.Linear(hidden_dim, 1),  # 输出一个 logit
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x).squeeze(-1)  # (B,)


class _TextLabelDataset(Dataset[Tuple[str, float]]):
	def __init__(self, texts: Sequence[Any], labels: np.ndarray) -> None:
		self.texts = [str(t) for t in texts]
		self.labels = labels.astype(np.float32)

	def __len__(self) -> int:
		return len(self.texts)

	def __getitem__(self, idx: int) -> Tuple[str, float]:
		return self.texts[idx], float(self.labels[idx])


def _freeze_bert_parameters(model: nn.Module) -> None:
	for p in model.parameters():
		p.requires_grad = False


def _unfreeze_last_n_encoder_layers(model: nn.Module, n: int) -> int:
	"""尽量通用地解冻最后 n 个 encoder layer；成功返回实际解冻的层数."""
	if n <= 0:
		return 0

	# BERT/RoBERTa/DeBERTa 常见结构：model.encoder.layer
	encoder = getattr(model, "encoder", None)
	if encoder is None:
		return 0
	layer_stack = getattr(encoder, "layer", None)
	if layer_stack is None:
		return 0

	# layer_stack 通常是 ModuleList
	try:
		layers = list(layer_stack)
	except Exception:
		return 0

	if not layers:
		return 0

	to_unfreeze = layers[-min(n, len(layers)) :]
	for layer in to_unfreeze:
		for p in layer.parameters():
			p.requires_grad = True
	return len(to_unfreeze)


class BertBinarySelector(BaseSelector):
	"""输入原始文本的二分类选择器：Frozen BERT + Trainable Head.

	训练信号：labels 为 0/1（例如：批内 KL top-k 记为 1，其余为 0）。
	打分输出：p(y=1|x)，可直接用于 top-k 选样。
	"""

	def __init__(
		self,
		model_name: str = "bert-base-uncased",
		max_length: int = 512,
		head_hidden_dim: int = 64,
		head_dropout: float = 0.1,
		lr: float = 1e-3,
		weight_decay: float = 0.0,
		freeze_bert: bool = True,
		unfreeze_last_n_layers: int = 0,
		device: Optional[str] = None,
	) -> None:
		if AutoModel is None or AutoTokenizer is None:
			raise ImportError(
				"transformers 未安装或不可用：BertBinarySelector 需要 transformers 包。"
			)

		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		# 
		# 设置为 left truncation 可以优先保留末尾（通常包含候选输出与关键上下文）。
		#
		try:
			self.tokenizer.truncation_side = "left"
		except Exception:
			pass
		self.bert = AutoModel.from_pretrained(model_name)
		self.bert.to(self.device)

		requested_max_length = int(max_length)
		limits: List[int] = [requested_max_length]

		# tokenizer.model_max_length 可能是一个极大哨兵值（表示“未知上限”），需过滤。
		tok_limit = getattr(self.tokenizer, "model_max_length", None)
		if isinstance(tok_limit, int) and tok_limit > 0 and tok_limit <= 1_000_000:
			limits.append(int(tok_limit))

		bert_limit = getattr(getattr(self.bert, "config", None), "max_position_embeddings", None)
		if isinstance(bert_limit, int) and bert_limit > 0:
			limits.append(int(bert_limit))

		self.max_length = int(min(limits))
		if self.max_length < requested_max_length:
			print(
				f"[BertBinarySelector] max_length={requested_max_length} 超过模型上限，"
				f"自动截断为 {self.max_length}。"
			)

		hidden_size = int(getattr(self.bert.config, "hidden_size", 768))
		self.head = _ClassificationHead(hidden_size, hidden_dim=head_hidden_dim, dropout=head_dropout).to(
			self.device
		)
		self.loss_fn = nn.BCEWithLogitsLoss()

		if freeze_bert:
			_freeze_bert_parameters(self.bert)
			_unfreeze_last_n_encoder_layers(self.bert, int(unfreeze_last_n_layers))

		# 只优化 requires_grad=True 的参数（默认仅 head；如果解冻部分层，也会被纳入）
		params = [p for p in list(self.bert.parameters()) + list(self.head.parameters()) if p.requires_grad]
		self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

		self._train_texts: List[str] = []
		self._train_labels: Optional[np.ndarray] = None

	def _to_text(self, x: Any) -> str:
		"""把输入样本转换成用于 selector 的文本。

		对 Judge prompt 做一个轻量预处理：如果包含 `### System ... ### User` 结构，
		则裁剪掉恒定的 System 区块，避免在较小 max_length 下信息量不足。
		"""
		t = str(x)
		# 常见结构："### System\n...\n\n### User\n..."
		if "### User" in t and t.lstrip().startswith("### System"):
			_, tail = t.split("### User", 1)
			t = "### User" + tail
		return t

	def _encode(self, texts: Sequence[str]) -> dict:
		enc = self.tokenizer(
			list(texts),
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		# Longformer needs global attention on the first token so its pooled
		# representation can aggregate the complete long sequence.
		model_type = str(getattr(getattr(self.bert, "config", None), "model_type", ""))
		if model_type == "longformer":
			global_attention_mask = torch.zeros_like(enc["input_ids"])
			global_attention_mask[:, 0] = 1
			enc["global_attention_mask"] = global_attention_mask
		return {k: v.to(self.device) for k, v in enc.items()}

	def _forward_logits(self, texts: Sequence[str]) -> torch.Tensor:
		batch = self._encode(texts)
		outputs = self.bert(**batch)
		# 优先使用 pooler_output（如果模型提供），否则退回 CLS 向量
		pool = getattr(outputs, "pooler_output", None)
		if pool is None:
			pool = outputs.last_hidden_state[:, 0, :]
		logits = self.head(pool)
		return logits

	def score(self, inputs: Sequence[Any]) -> np.ndarray:
		texts = [self._to_text(x) for x in inputs]
		self.head.eval()
		# 若 BERT 冻结则用 eval（更稳定且更快），否则也需要 eval 来做推理
		self.bert.eval()
		
		# 修改：使用 mini-batch 推理，防止 unlabeled pool 过大导致 OOM
		# Long-context encoders have quadratic-ish activation pressure in the
		# sequence length; keep their inference batches deliberately small.
		inference_batch_size = 2 if self.max_length > 1024 else 32
		all_probs = []

		with torch.no_grad():
			for i in range(0, len(texts), inference_batch_size):
				batch_texts = texts[i : i + inference_batch_size]
				logits = self._forward_logits(batch_texts)
				probs = torch.sigmoid(logits).detach().cpu().numpy()
				all_probs.append(probs)
		
		if not all_probs:
			return np.array([], dtype=np.float32)
		return np.concatenate(all_probs, axis=0).astype(np.float32)

	def update(
		self,
		inputs: Sequence[Any],
		labels: np.ndarray,
		epochs: int = 1,
		batch_size: int = 16,
	) -> None:
		labels = labels.astype(np.float32)
		texts = [self._to_text(x) for x in inputs]

		if self._train_labels is None:
			self._train_texts = list(texts)
			self._train_labels = labels
		else:
			self._train_texts.extend(texts)
			self._train_labels = np.concatenate([self._train_labels, labels], axis=0)

		dataset = _TextLabelDataset(self._train_texts, self._train_labels)

		def collate_fn(batch_items: List[Tuple[str, float]]) -> Tuple[dict, torch.Tensor]:
			batch_texts = [t for t, _ in batch_items]
			batch_labels = torch.tensor([y for _, y in batch_items], dtype=torch.float32, device=self.device)
			enc = self._encode(batch_texts)
			return enc, batch_labels

		loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

		# 训练模式：如果 BERT 全冻结，保持 eval 也可以；否则需要 train
		self.head.train()
		any_bert_trainable = any(p.requires_grad for p in self.bert.parameters())
		self.bert.train(mode=any_bert_trainable)

		for _ in range(int(epochs)):
			for enc, yb in loader:
				outputs = self.bert(**enc)
				pool = getattr(outputs, "pooler_output", None)
				if pool is None:
					pool = outputs.last_hidden_state[:, 0, :]
				logits = self.head(pool)
				loss = self.loss_fn(logits, yb)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

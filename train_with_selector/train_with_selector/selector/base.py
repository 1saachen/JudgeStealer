from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import numpy as np


class BaseSelector(ABC):
	"""选择器抽象基类.

	负责：
	- 根据样本特征打分（score）
	- 根据真实 KL 改善信号更新自身（update）

	不限制使用什么训练方法，方便后续扩展。
	"""

	@abstractmethod
	def score(self, inputs: Sequence[Any]) -> np.ndarray:
		"""对一批原始样本打分（如文本）.

		返回 shape = (num_samples,) 的实数分数，越大表示越值得查询。
		"""

	@abstractmethod
	def update(self, inputs: Sequence[Any], kl_values: np.ndarray, **kwargs: Any) -> None:
		"""使用新的 (输入, 真实 KL 改善/伪标签) 对选择器进行训练/微调.

		说明：不同 selector 的训练方式不同。
		- 基于神经网络的 selector 可能支持 epochs/batch_size 等参数；
		- RandomSelector 等不需要训练的实现应忽略这些参数。
		"""

	def get_state(self) -> Dict[str, Any]:  # 可选：方便保存/恢复
		return {}

	def load_state(self, state: Dict[str, Any]) -> None:
		pass

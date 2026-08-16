from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence, Tuple

import numpy as np


class VictimModel(ABC):
	"""受害者模型接口.

	你可以用 HTTP 调用、vLLM、本地 LLM 等方式实现具体子类，
	只要满足 query 的签名即可。
	"""

	@abstractmethod
	def query(self, inputs: Sequence[Any]) -> np.ndarray:
		"""查询受害者模型.

		参数
		------
		inputs: 一批原始样本（可以是文本、特征向量等），由你在具体实现中解释。

		返回
		------
		labels: np.ndarray[int] 或其它离散标签编码（硬标签）。
		"""


class ProxyModel(ABC):
	"""代理模型接口.

	框架只依赖这个抽象类，不依赖具体实现，便于后续换模型。
	"""

	@abstractmethod
	def predict_proba(self, inputs: Sequence[Any]) -> np.ndarray:
		"""返回代理模型对每个样本的预测分布 p(y|x).

		要求返回形状为 (batch_size, num_classes) 的概率矩阵。
		用于 KL 散度计算和特征抽取。
		"""

	@abstractmethod
	def train_on_batch(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
		"""在一个 batch 的新标注上更新代理模型.

		具体实现可以是微调 LLM、训练分类器等。
		"""

	@abstractmethod
	def clone(self) -> "ProxyModel":
		"""返回当前代理模型的一个 *轻量* 拷贝.

		用于需要对比训练前 / 后分布时（例如更精细的 KL 模拟）。
		如果代价太大，你也可以在具体实现里抛出 NotImplementedError，
		然后在上层选择相应的近似方法。
		"""


def batch_predict_kl_inputs(
	model: ProxyModel, inputs: Sequence[Any]
) -> Tuple[np.ndarray, np.ndarray]:
	"""辅助函数：给定模型和样本，返回 (logits 或 probs, probs).

	目前假设 predict_proba 已经输出的是概率分布，这里简单复用；
	如果你之后需要 logits，可以在你自己的 ProxyModel 子类里扩展接口。
	"""

	probs = model.predict_proba(inputs)
	return probs, probs

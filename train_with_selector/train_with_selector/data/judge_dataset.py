from __future__ import annotations

"""Judge 数据集加载与 prompt 构造。

支持两种常见格式（JSON list）：

1) **Raw 记录 + models 列表**（默认格式）

{
	"id": 1,
	"dataset": "...",
	"Instruction": "...",
	"input": "...",
	"reference": "...",
	"models": [
		{"model": "Qwen...", "output": "...", "score": 1..10},
		...
	]
}

2) **Flattened 扁平样本**（每条就是一个候选 output）

{
	"id": 1,
	"dataset": "...",
	"Instruction": "...",
	"input": "...",
	"model": "Qwen...",
	"output": "...",
	"reason": "...",   # 可选：可用于 COT 监督（reason -> Score）训练
	"score": 1..10
}

本模块会把每个候选输出展平为一条可训练样本：
- text: 拼好的 judge prompt（包含 system prompt + instruction/input + candidate output）
- label: 分数映射到分类标签 (0..num_labels-1)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import json


JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION = """You are an impartial judge evaluating AI assistant responses.

Your evaluation must consider:
- Helpfulness
- Relevance
- Accuracy
- Depth
- Creativity
- Level of detail

Provide a brief explanation, then give ONLY a numeric score 1-10 in this EXACT format:
Score: [X]

Example:
The answer is accurate and helpful but lacks depth.
Score: [7]
"""

JUDGE_SYSTEM_PROMPT_SCORE_ONLY = """You are an impartial judge evaluating AI assistant responses.

Your evaluation must consider:
- Helpfulness
- Relevance
- Accuracy
- Depth
- Creativity
- Level of detail

Give ONLY a numeric score 1-10 in this EXACT format:
Score: [X]

Example:
The answer is accurate and helpful but lacks depth.
Score: [7]
"""

DEFAULT_JUDGE_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION

JUDGE_SYSTEM_PROMPT_VARIANTS: Dict[str, str] = {
	"explanation": JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION,
	"with_explanation": JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION,
	"current": JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION,
	"score_only": JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
	"no_explanation": JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
}


def get_judge_system_prompt(version: str = "explanation") -> str:
	"""Return one of the supported judge system prompt variants."""
	key = str(version or "explanation").strip().lower().replace("-", "_")
	try:
		return JUDGE_SYSTEM_PROMPT_VARIANTS[key]
	except KeyError as exc:
		valid = ", ".join(sorted(JUDGE_SYSTEM_PROMPT_VARIANTS))
		raise ValueError(f"unknown judge prompt version {version!r}; choices: {valid}") from exc


def score_to_class(score: int, score_min: int = 1, score_max: int = 10) -> int:
	"""把打分 (1..10) 映射为分类标签 (0..9)。"""
	if score < score_min or score > score_max:
		raise ValueError(f"score must be within [{score_min}, {score_max}], got {score}")
	return int(score - score_min)


def class_to_score(label: int, score_min: int = 1, score_max: int = 10) -> int:
	"""把分类标签 (0..9) 还原为打分 (1..10)。"""
	if label < 0 or label > (score_max - score_min):
		raise ValueError(
			f"label must be within [0, {score_max - score_min}], got {label}"
		)
	return int(label + score_min)


def build_judge_prompt(
	*,
	system_prompt: str,
	instruction: str,
	input_text: str,
	candidate_output: str,
	gold_score: Optional[int] = None,
	include_gold_score: bool = False,
	fix_score_prefix: bool = True,
) -> str:
	"""构造喂给 judge 的纯文本 prompt。

	参数：
	- fix_score_prefix: 是否在末尾固定添加 "Score: ["
	  * True: prompt 以 "Score: [" 结尾，模型直接生成分数
	  * False: prompt 不包含 "Score: ["，模型需要自己生成完整格式

	说明：
	- 这里用纯文本拼接，便于兼容 `str(x)` 作为下游输入。
	- 如果你后续要走 chat template（tokenizer.apply_chat_template），
	  可以把这里换成 messages 结构，并在具体模型侧做模板化。
	"""
	instruction = (instruction or "").strip()
	input_text = (input_text or "").strip()
	candidate_output = (candidate_output or "").strip()

	parts: List[str] = ["### System", system_prompt.strip(), "", "### User"]
	parts.append(f"Instruction: {instruction}")
	if input_text:
		parts.append(f"Input: {input_text}")
	parts.append("")
	parts.append("### Assistant")
	parts.append(candidate_output)
	parts.append("")
	parts.append("### Judge")
	if include_gold_score and gold_score is not None:
		# 仅用于消融/上限实验：在真实未标注选择时不可用（会泄漏标签）。
		parts.append(f"GoldScore: {int(gold_score)}")
	parts.append("Please evaluate the assistant response and output in the required format.")
	parts.append("")
	
	# 可选：在末尾固定添加 "Score: ["
	if fix_score_prefix:
		parts.append("Score: [")
	
	return "\n".join(parts)


@dataclass(frozen=True)
class JudgeExample:
	"""一条可训练/可查询样本。

	- `__str__` 返回 prompt 文本，确保 selector/proxy 直接 `str(x)` 即可。
	- `label` 是分类标签（0..num_labels-1）。
	"""

	id: int
	dataset: str
	model: str
	prompt: str
	reason: str
	score: int
	label: int

	def __str__(self) -> str:  # noqa: D105
		return self.prompt


@dataclass(frozen=True)
class MultiDimJudgeExample:
	"""多维度打分数据的一条可训练/可查询样本。

	与 JudgeExample 一致：
	- `__str__` 返回 prompt 文本
	- `label` 为分类标签（0..num_labels-1）

	额外字段：
	- dimension: 维度名（如 Helpfulness/Accuracy/...）
	- group_id: 原始样本 id，用于 query_unit='group' 时把同一条原始记录的多个维度视为一组
	"""

	id: int
	group_id: int
	dataset: str
	model: str
	dimension: str
	prompt: str
	reason: str
	score: int
	label: int

	def __str__(self) -> str:  # noqa: D105
		return self.prompt


def _safe_int(x: Any, default: int = -1) -> int:
	try:
		return int(x)
	except Exception:
		return default


def load_judge_json(
	path: str,
	*,
	system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
	score_min: int = 1,
	score_max: int = 10,
	model_index: Optional[int] = None,
	include_gold_score_in_prompt: bool = False,
	fix_score_prefix_in_prompt: bool = True,
) -> List[JudgeExample]:
	"""从 JSON(list) 读取并展平为 JudgeExample 列表。

	参数：
	- model_index: 如果只想取 `models` 中某一个候选（例如只取第 0 个），传入索引。
	  默认 None 表示把 `models` 全部展平。
	- fix_score_prefix_in_prompt: 是否在 prompt 末尾固定添加 "Score: ["
	"""
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		raise ValueError("Judge dataset JSON must be a list of records.")

	examples: List[JudgeExample] = []
	for rec in data:
		if not isinstance(rec, dict):
			continue
		rec_id = _safe_int(rec.get("id", -1), default=-1)
		dataset = str(rec.get("dataset", ""))
		instruction = str(rec.get("Instruction", rec.get("instruction", "")))
		input_text = str(rec.get("input", ""))

		# 兼容两种数据格式：
		# - raw 格式：rec["models"] 是 list
		# - flattened 格式：rec 直接包含 model/output/score（把它当成单元素 models）
		models = rec.get("models", None)
		if isinstance(models, list) and models:
			pass
		else:
			flat_model = rec.get("model", None)
			flat_output = rec.get("output", None)
			flat_score = rec.get("score", None)
			if flat_model is None or flat_output is None or flat_score is None:
				continue
			flat_reason = rec.get("reason", "")
			models = [{"model": flat_model, "output": flat_output, "score": flat_score, "reason": flat_reason}]

		chosen_models: Iterable[Dict[str, Any]]
		if model_index is None:
			chosen_models = [m for m in models if isinstance(m, dict)]
		else:
			if 0 <= int(model_index) < len(models) and isinstance(models[int(model_index)], dict):
				chosen_models = [models[int(model_index)]]
			else:
				chosen_models = []

		for m in chosen_models:
			model_name = str(m.get("model", ""))
			output = str(m.get("output", ""))
			reason = str(m.get("reason", rec.get("reason", "")) or "")
			score = _safe_int(m.get("score", -1), default=-1)
			if score < score_min or score > score_max:
				continue
			label = score_to_class(score, score_min=score_min, score_max=score_max)
			prompt = build_judge_prompt(
				system_prompt=system_prompt,
				instruction=instruction,
				input_text=input_text,
				candidate_output=output,
				gold_score=score,
				include_gold_score=bool(include_gold_score_in_prompt),
				fix_score_prefix=bool(fix_score_prefix_in_prompt),
			)
			examples.append(
				JudgeExample(
					id=rec_id,
					dataset=dataset,
					model=model_name,
					prompt=prompt,
					reason=reason,
					score=score,
					label=label,
				)
			)

	return examples


def load_judge_json_multidim(
	path: str,
	*,
	system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
	score_min: int = 1,
	score_max: int = 10,
	dimensions: Optional[Sequence[str]] = None,
	include_gold_score_in_prompt: bool = False,
	fix_score_prefix_in_prompt: bool = True,
	append_dimension_to_system_prompt: bool = True,
) -> List[MultiDimJudgeExample]:
	"""从包含 `scores: {dim: score}` 的 JSON(list) 读取并展平为 MultiDimJudgeExample。

	兼容的数据格式（每条 record 是一个 candidate output）：
	{
	  "id": 1,
	  "dataset": "...",
	  "Instruction": "...",
	  "input": "...",
	  "model": "...",
	  "output": "...",
	  "scores": {"Helpfulness": 9, "Accuracy": 8, ...}
	}

	展平策略：对每个 (dimension, score) 生成一条样本；默认 group_id=id。
	"""
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		raise ValueError("Multi-dim judge dataset JSON must be a list of records.")

	dim_allow: Optional[set[str]] = None
	if dimensions is not None:
		dim_allow = {str(d).strip() for d in dimensions if str(d).strip()}
		if not dim_allow:
			dim_allow = None

	examples: List[MultiDimJudgeExample] = []
	for rec in data:
		if not isinstance(rec, dict):
			continue
		rec_id = _safe_int(rec.get("id", -1), default=-1)
		dataset = str(rec.get("dataset", ""))
		instruction = str(rec.get("Instruction", rec.get("instruction", "")))
		input_text = str(rec.get("input", ""))
		model_name = str(rec.get("model", ""))
		output = str(rec.get("output", ""))
		reason = str(rec.get("reason", "") or "")

		scores = rec.get("scores", None)
		if not isinstance(scores, dict) or not scores:
			continue

		group_id = _safe_int(rec.get("group_id", rec_id), default=rec_id)

		for dim, score_v in scores.items():
			dim_name = str(dim)
			if dim_allow is not None and dim_name not in dim_allow:
				continue
			score = _safe_int(score_v, default=-1)
			if score < score_min or score > score_max:
				continue
			label = score_to_class(score, score_min=score_min, score_max=score_max)
			sys_p = str(system_prompt).strip()
			if append_dimension_to_system_prompt:
				sys_p = sys_p + "\n\n" + (
					"IMPORTANT: You are scoring ONLY the following dimension: " + dim_name + "."
				)
			prompt = build_judge_prompt(
				system_prompt=sys_p,
				instruction=instruction,
				input_text=input_text,
				candidate_output=output,
				gold_score=score,
				include_gold_score=bool(include_gold_score_in_prompt),
				fix_score_prefix=bool(fix_score_prefix_in_prompt),
			)
			examples.append(
				MultiDimJudgeExample(
					id=rec_id,
					group_id=group_id,
					dataset=dataset,
					model=model_name,
					dimension=dim_name,
					prompt=prompt,
					reason=reason,
					score=score,
					label=label,
				)
			)

	return examples

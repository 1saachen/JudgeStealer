from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import importlib
import inspect

import numpy as np
import torch
from torch import nn

from .base import ProxyModel


class LlamaProxyModel(ProxyModel):
	"""用本地 Llama-7B-Instruct 作为 *冻结特征提取器* 的代理分类模型。

	设计目标：
	- 兼容框架的 `ProxyModel` 接口（`predict_proba` / `train_on_batch`）。
	- 采用参数高效微调（LoRA），避免全量微调带来的巨大显存/速度开销。
	- 支持本地模型目录（包含 tokenizer/model 权重）加载。

	输入假设：
	- `inputs` 是 `Sequence[str]`（或可被 `str(x)` 转成文本的对象）。
	- `labels` 是 int 类别 id（0..num_labels-1）。
	"""

	def __init__(
		self,
		model_path: str,
		num_labels: int = 2,
		lr: float = 1e-3,
		weight_decay: float = 0.0,
		max_length: int = 128,
		max_new_tokens: int = 64,
		device: Optional[str] = None,
		predict_mode: str = "classifier",
		anchor_score_prefix: bool = False,
		reason_max_tokens: int = 0,
		ignore_unparsable_score: bool = False,
		score_min: int = 1,
		score_max: int = 10,
		torch_dtype: str = "auto",
		trust_remote_code: bool = False,
		finetune_mode: str = "lora",
		gradient_checkpointing: bool = True,
		use_amp: bool = True,
		lora_r: int = 8,
		lora_alpha: int = 16,
		lora_dropout: float = 0.05,
		lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
		load_in_4bit: bool = False,
		fix_score_prefix_in_prompt: bool = True,
		multidim_dimensions: Optional[Sequence[str]] = None,
		pointwise_loss_type: str = "ce",
		pointwise_distance_weight: float = 0.0,
	) -> None:
		try:
			transformers = importlib.import_module("transformers")
			extractions = (getattr(transformers, "AutoTokenizer", None), getattr(transformers, "AutoModelForCausalLM", None))
			if extractions[0] is None or extractions[1] is None:
				raise ImportError("transformers is missing AutoTokenizer/AutoModelForCausalLM")
			AutoTokenizer = extractions[0]
			AutoModelForCausalLM = extractions[1]
		except Exception as e:  # pragma: no cover
			raise ImportError(
				"transformers 未安装或不可用：LlamaProxyModel 需要 transformers 包。"
			) from e
		self.model_path = model_path
		self.num_labels = int(num_labels)
		if self.num_labels <= 1:
			raise ValueError(f"num_labels must be >= 2, got {num_labels}")

		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.max_length = int(max_length)
		self.max_new_tokens = int(max_new_tokens)
		if self.max_new_tokens <= 0:
			self.max_new_tokens = 1
		self.predict_mode = str(predict_mode)
		if self.predict_mode not in {"classifier", "lm_head_scores", "lm_head_multidim_scores"}:
			raise ValueError(
				"predict_mode must be 'classifier', 'lm_head_scores' or 'lm_head_multidim_scores', got "
				+ str(predict_mode)
			)
		self.multidim_dimensions = list(multidim_dimensions) if multidim_dimensions is not None else None
		self.score_min = int(score_min)
		self.score_max = int(score_max)
		if self.score_min > self.score_max:
			raise ValueError(f"score_min must be <= score_max, got {score_min}>{score_max}")
		self.finetune_mode = str(finetune_mode)
		if self.finetune_mode not in {"lora", "full"}:
			raise ValueError(
				"finetune_mode must be 'lora' or 'full', got " + str(finetune_mode)
			)
		if self.finetune_mode == "full" and bool(load_in_4bit):
			raise ValueError(
				"finetune_mode='full' is not compatible with load_in_4bit=True. "
				"Disable 4bit quantization or use finetune_mode='lora'."
			)
		# AMP：
		# - fp16: 需要 GradScaler（否则极易 NaN）
		# - bf16: 不需要/不应使用 GradScaler（部分 torch 版本对 bf16 unscale 不支持）
		# - fp32: 关闭 AMP
		self.use_amp = bool(use_amp) and (self.device.type == "cuda")
		self._amp_dtype: Optional[torch.dtype] = None
		self.lora_r = int(lora_r)
		self.lora_alpha = int(lora_alpha)
		self.lora_dropout = float(lora_dropout)
		self.lora_target_modules = str(lora_target_modules)

		# 是否在 prompt 末尾固定添加 "Score: ["
		# True: prompt 已包含 "Score: ["，模型直接生成分数（推荐）
		# False: prompt 不包含，模型需要自己生成完整的 "Score: [X]" 格式
		self.fix_score_prefix_in_prompt = bool(fix_score_prefix_in_prompt)
		# 在 fix_score_prefix_in_prompt=False 时的兜底：
		# 不依赖模型生成出 "Score: ["，而是在推理时临时把 prompt 末尾锚定到 "Score: ["，
		# 直接用第一个生成 step 的 logits 计算 p(score|x)。
		self.anchor_score_prefix = bool(anchor_score_prefix)
		# CoT(reason) 监督时最多保留的 reason token 数（0 表示不限制）
		self.reason_max_tokens = int(reason_max_tokens)
		# 解析不到 `Score: [` 时是否返回 NaN（由上层 pipeline 忽略）
		self.ignore_unparsable_score = bool(ignore_unparsable_score)
		self.pointwise_class_weights: Optional[torch.Tensor] = None
		self.pointwise_loss_type = str(pointwise_loss_type)
		if self.pointwise_loss_type not in {"ce", "ce_mse", "ce_cost", "ordinal", "coral"}:
			raise ValueError(
				"pointwise_loss_type must be one of {'ce','ce_mse','ce_cost','ordinal','coral'}, got "
				+ str(pointwise_loss_type)
			)
		self.pointwise_distance_weight = float(pointwise_distance_weight)
		if self.pointwise_distance_weight < 0.0:
			raise ValueError("pointwise_distance_weight must be >= 0")

		self.tokenizer = AutoTokenizer.from_pretrained(
			model_path,
			use_fast=True,
			trust_remote_code=trust_remote_code,
			padding_side="left",
		)
		# 重要：统一使用左侧截断，保留结尾的 `Score: [` 等关键信息
		# 这样在 predict_proba 等推理阶段也不会把尾部提示截掉
		setattr(self.tokenizer, "truncation_side", "left")
		# Llama 系列常见：没有 pad_token；这里用 eos 作为 padding。
		if self.tokenizer.pad_token_id is None:
			if self.tokenizer.eos_token_id is None:
				raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; cannot pad.")
			self.tokenizer.pad_token = self.tokenizer.eos_token
			self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

		resolved_dtype: Optional[torch.dtype]
		if torch_dtype == "auto":
			# full finetune 用 fp16 很容易数值不稳定（尤其本项目是在线/小步更新），
			# 优先用 bf16（若硬件支持），否则退回 fp32。
			if self.device.type == "cuda" and self.finetune_mode == "full":
				try:
					if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
						resolved_dtype = torch.bfloat16
					else:
						resolved_dtype = torch.float32
				except Exception:
					resolved_dtype = torch.float32
			else:
				resolved_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
		else:
			try:
				resolved_dtype = getattr(torch, torch_dtype)
			except AttributeError as e:
				raise ValueError(
					f"Invalid torch_dtype={torch_dtype}. Use 'auto' or a torch dtype name like 'float16'."
				) from e

		# 根据权重 dtype 决定 autocast dtype
		if self.use_amp:
			if resolved_dtype == torch.float16:
				self._amp_dtype = torch.float16
			elif resolved_dtype == torch.bfloat16:
				self._amp_dtype = torch.bfloat16
			else:
				# float32 或其它：不启用 autocast
				self.use_amp = False
				self._amp_dtype = None

		# transformers 新版本提示 torch_dtype 已废弃，改用 dtype；这里做兼容。
		model_kwargs: Dict[str, Any] = {
			"low_cpu_mem_usage": True,
			"trust_remote_code": trust_remote_code,
		}
		
		# 4-bit 量化支持
		if load_in_4bit:
			try:
				from transformers import BitsAndBytesConfig
				quantization_config = BitsAndBytesConfig(
					load_in_4bit=True,
					bnb_4bit_compute_dtype=resolved_dtype,  # 用 bf16/fp16 计算
					bnb_4bit_use_double_quant=True,
					bnb_4bit_quant_type="nf4",
				)
				model_kwargs["quantization_config"] = quantization_config
				# 4bit quantization requires device_map="auto" usually, 
				# but we can try letting transformers handle it or specifying specific device.
				# If we don't specify device_map, from_pretrained might put it on CPU first?
				# Usually for qlora we use device_map="auto" or specific device.
				# We already have self.device. Let's try to not force device_map="auto" if user didn't ask, 
				# but bitsandbytes usually needs it to decide placement.
				# However, since we act as a library, let's keep it simple.
				if "device_map" not in model_kwargs:
					# Explicitly put on current device if possible, or auto
					model_kwargs["device_map"] = {"": self.device.index if self.device.index is not None else 0}
			except ImportError:
				print("WARNING: bitsandbytes not found or transformers too old. Ignoring load_in_4bit=True.")
			except Exception as e:
				print(f"WARNING: Failed to setup quantization config: {e}")

		# Try to use 'dtype' first as 'torch_dtype' is deprecated in newer transformers
		try:
			sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
			if "dtype" in sig.parameters:
				model_kwargs["dtype"] = resolved_dtype
			elif "torch_dtype" in sig.parameters:
				model_kwargs["torch_dtype"] = resolved_dtype
			else:
				# Fallback
				model_kwargs["torch_dtype"] = resolved_dtype
		except Exception:
			# If inspection fails, just use dtype as it is the future standard
			model_kwargs["dtype"] = resolved_dtype
		
		self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
		
		# 如果没有量化，则显式移动到 device；如果用了 device_map (量化时)，则不需要手动 to(device)
		if "device_map" not in model_kwargs:
			self.model.to(self.device)
			
		# Prepare for k-bit training if quantized (only meaningful for LoRA/QLoRA)
		if load_in_4bit:
			from peft import prepare_model_for_kbit_training
			self.model = prepare_model_for_kbit_training(
				self.model,
				use_gradient_checkpointing=bool(gradient_checkpointing),
			)

		if self.finetune_mode == "lora":
			# 训练策略：参数高效微调（LoRA）
			try:
				peft = importlib.import_module("peft")
				extractions = (
					getattr(peft, "LoraConfig", None),
					getattr(peft, "get_peft_model", None),
					getattr(peft, "TaskType", None),
				)
				if extractions[0] is None or extractions[1] is None or extractions[2] is None:
					raise ImportError("peft is missing LoraConfig/get_peft_model/TaskType")
				LoraConfig, get_peft_model, TaskType = extractions
			except Exception as e:  # pragma: no cover
				raise ImportError(
					"finetune_mode='lora' 需要额外依赖 peft。请先安装：pip install peft"
				) from e

			for p in self.model.parameters():
				p.requires_grad_(False)
			targets = [s.strip() for s in self.lora_target_modules.split(",") if s.strip()]
			if not targets:
				targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
			lora_cfg = LoraConfig(
				task_type=TaskType.CAUSAL_LM,
				r=max(1, self.lora_r),
				lora_alpha=max(1, self.lora_alpha),
				lora_dropout=max(0.0, self.lora_dropout),
				target_modules=targets,
				bias="none",
			)
			self.model = get_peft_model(self.model, lora_cfg)
			self.model.to(self.device)
			self.model.train()

			if bool(gradient_checkpointing) and hasattr(self.model, "gradient_checkpointing_enable"):
				self.model.gradient_checkpointing_enable()
				if hasattr(self.model, "config"):
					self.model.config.use_cache = False
				# transformers 的 gradient checkpointing 需要至少一个输入张量 requires_grad=True。
				# LoRA 场景下若未开启，会导致梯度为 None，最终 loss 不带 grad_fn。
				if hasattr(self.model, "enable_input_require_grads"):
					self.model.enable_input_require_grads()

			# 确认 LoRA 确实挂上了可训练参数（target_modules 不匹配时可能导致没有任何 requires_grad）。
			trainable_names = [
				n for n, p in self.model.named_parameters() if getattr(p, "requires_grad", False)
			]
			if not trainable_names:
				raise ValueError(
					"LoRA setup produced 0 trainable parameters. This usually means your "
					"lora_target_modules do not match the model's module names. "
					"Try setting llama_lora_target_modules to the correct projection layers."
				)
		else:
			# finetune_mode == 'full'
			for p in self.model.parameters():
				p.requires_grad_(True)
			self.model.to(self.device)
			self.model.train()
			if bool(gradient_checkpointing) and hasattr(self.model, "gradient_checkpointing_enable"):
				self.model.gradient_checkpointing_enable()
				if hasattr(self.model, "config"):
					self.model.config.use_cache = False

		# --------- 两种预测模式 ---------
		# 1) classifier: 取 hidden_state 特征 + 训练分类头
		# 2) lm_head_scores: 用 LM head 的 vocab logits 对分数 token 计算 p(score|x)
		if self.predict_mode == "classifier":
			hidden_size = (
				getattr(self.model.config, "hidden_size", None)
				or getattr(self.model.config, "n_embd", None)
			)
			if hidden_size is None:
				raise ValueError(
					"Cannot infer hidden size from model config (hidden_size / n_embd missing)."
				)
			self.hidden_size = int(hidden_size)
			self.classifier: Optional[nn.Module] = nn.Linear(self.hidden_size, self.num_labels).to(self.device)
			# 优化所有 requires_grad=True 的参数（LoRA adapters 或 full 模式的全参）+ 分类头
			trainable_params = [
				p
				for p in list(self.model.parameters()) + list(self.classifier.parameters())
				if p.requires_grad
			]
			try:
				self.optimizer = torch.optim.AdamW(
					trainable_params, lr=lr, weight_decay=weight_decay, foreach=False
				)
			except TypeError:
				self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
			self.loss_fn = nn.CrossEntropyLoss()
			self._score_token_map: Optional[Dict[int, dict]] = None
			self._score_values: Optional[List[int]] = None
		elif self.predict_mode == "lm_head_scores":
			# lm_head_scores 模式：要求标签数与 score range 一致
			expected = self.score_max - self.score_min + 1
			if self.num_labels != expected:
				raise ValueError(
					f"lm_head_scores requires num_labels == (score_max-score_min+1). "
					f"Got num_labels={self.num_labels}, expected={expected}."
				)
			# LoRA 会更新模型行为，因此允许。
			self.hidden_size = 0
			self.classifier = None
			# 只优化可训练参数（LoRA adapter）
			trainable_params = [p for p in self.model.parameters() if p.requires_grad]
			try:
				self.optimizer = torch.optim.AdamW(
					trainable_params, lr=lr, weight_decay=weight_decay, foreach=False
				)
			except TypeError:
				self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
			self.loss_fn = nn.CrossEntropyLoss()
			self._score_values = list(range(self.score_min, self.score_max + 1))
			self._score_token_map = self.get_tokens_for_scores(self._score_values)
		else:
			# lm_head_multidim_scores: same score token map as lm_head_scores, but returns (B, D, K)
			expected = self.score_max - self.score_min + 1
			if self.num_labels != expected:
				raise ValueError(
					f"lm_head_multidim_scores requires num_labels == (score_max-score_min+1). "
					f"Got num_labels={self.num_labels}, expected={expected}."
				)
			if not self.multidim_dimensions:
				raise ValueError("lm_head_multidim_scores requires multidim_dimensions to be provided.")
			self.hidden_size = 0
			self.classifier = None
			trainable_params = [p for p in self.model.parameters() if p.requires_grad]
			try:
				self.optimizer = torch.optim.AdamW(
					trainable_params, lr=lr, weight_decay=weight_decay, foreach=False
				)
			except TypeError:
				self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
			self.loss_fn = nn.CrossEntropyLoss()
			self._score_values = list(range(self.score_min, self.score_max + 1))
			self._score_token_map = self.get_tokens_for_scores(self._score_values)
			# Precompute open-template token ids and per-dimension bracket token positions.
			# IMPORTANT: Do NOT try to locate each prefix by matching separately-tokenized
			# substrings inside the tokenized full template. Some tokenizers (e.g., unigram)
			# may tokenize a substring differently depending on surrounding context.
			self._md_template_open_text = "\n".join([f"{d}: [" for d in self.multidim_dimensions])
			self._md_bracket_pos_in_template = []
			try:
				enc = self.tokenizer(
					self._md_template_open_text,
					add_special_tokens=False,
					return_offsets_mapping=True,
				)
				self._md_template_open_ids = enc.get("input_ids", [])
				offsets = enc.get("offset_mapping", [])
				# Compute the character position of each '[' in the template text.
				bracket_char_pos: List[int] = []
				cur = 0
				for i, d in enumerate(self.multidim_dimensions):
					line = f"{d}: ["
					if i > 0:
						cur += 1  # newline
					bracket_char_pos.append(cur + len(line) - 1)
					cur += len(line)

				# Map char positions to token indices using offsets.
				for cp in bracket_char_pos:
					pos_tok = -1
					for ti, (s, e) in enumerate(offsets):
						# offsets for special tokens may be (0,0)
						if int(s) <= int(cp) < int(e):
							pos_tok = int(ti)
							break
					if pos_tok < 0:
						raise ValueError(f"Failed to map bracket char pos to token index: {cp}")
					self._md_bracket_pos_in_template.append(pos_tok)
			except Exception:
				# Fallback: still keep template ids for suffix matching; bracket positions empty.
				# In this case, multidim prediction will fallback to uniform (safe but low quality).
				self._md_template_open_ids = self.tokenizer(
					self._md_template_open_text,
					add_special_tokens=False,
				).get("input_ids", [])
				self._md_bracket_pos_in_template = []
		# GradScaler 只用于 fp16；bf16 不启用 scaler（否则会触发 unscale bf16 的实现缺失）
		self._scaler: Optional[Any] = None
		if self.use_amp and self._amp_dtype == torch.float16:
			if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
				self._scaler = torch.amp.GradScaler("cuda")
			else:
				self._scaler = torch.cuda.amp.GradScaler()

	def _autocast_ctx(self):
		"""返回 autocast 上下文（根据 _amp_dtype）。"""
		if not self.use_amp or self._amp_dtype is None:
			return None
		# torch.amp.autocast / torch.cuda.amp.autocast 两者都兼容的写法
		if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
			return torch.amp.autocast("cuda", dtype=self._amp_dtype)
		return torch.cuda.amp.autocast(dtype=self._amp_dtype)

	def _encode(self, inputs: Sequence[Any]) -> dict:
		texts: List[str] = [str(x) for x in inputs]
		encoded = self.tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		return {k: v.to(self.device) for k, v in encoded.items()}

	def _get_backbone_for_hidden_states(self) -> nn.Module:
		"""Return the transformer backbone without materializing LM-head logits."""
		base_model: nn.Module = self.model
		get_base_model = getattr(base_model, "get_base_model", None)
		if callable(get_base_model):
			try:
				candidate = get_base_model()
				if isinstance(candidate, nn.Module):
					base_model = candidate
			except Exception:
				pass

		backbone = getattr(base_model, "model", None)
		if isinstance(backbone, nn.Module):
			return backbone
		return base_model

	def _forward_last_hidden_state(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
		"""Forward only through the backbone to avoid allocating vocab-sized logits."""
		backbone = self._get_backbone_for_hidden_states()
		outputs = backbone(
			**batch,
			use_cache=False,
			output_hidden_states=False,
			return_dict=True,
		)
		last_hidden_state = getattr(outputs, "last_hidden_state", None)
		if isinstance(last_hidden_state, torch.Tensor):
			return last_hidden_state

		hidden_states = getattr(outputs, "hidden_states", None)
		if hidden_states:
			return hidden_states[-1]

		if isinstance(outputs, (tuple, list)) and outputs:
			first = outputs[0]
			if isinstance(first, torch.Tensor):
				return first

		raise RuntimeError("Failed to obtain last hidden state from the transformer backbone.")

	def _pool_last_token_features(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor],
	) -> torch.Tensor:
		"""Pool the final non-padding token for classifier-style features."""
		if self.tokenizer.padding_side == "left":
			return hidden_states[:, -1, :]

		if attention_mask is None:
			return hidden_states[:, -1, :]

		last_indices = attention_mask.sum(dim=1) - 1
		last_indices = last_indices.clamp(min=0)
		batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
		return hidden_states[batch_indices, last_indices, :]

	def _encode_one_keep_suffix(self, text: str) -> dict:
		"""编码单条文本，并在超长时保留末尾 suffix。

		在本项目里我们把 prompt 结尾固定到 `Score: [`，因此分数相关 token 在末尾；
		默认 tokenizer 的 truncation 会保留前缀、丢掉末尾，反而会把 `Score: [` 截没。
		这里改为：超长时保留最后 max_length 个 token。
		"""
		enc = self.tokenizer(
			text,
			return_tensors="pt",
			add_special_tokens=True,
			truncation=False,
		)
		input_ids: torch.Tensor = enc["input_ids"].to(self.device)
		attention_mask: Optional[torch.Tensor] = enc.get("attention_mask")
		if attention_mask is not None:
			attention_mask = attention_mask.to(self.device)

		if input_ids.shape[1] > self.max_length:
			input_ids = input_ids[:, -self.max_length :]
			if attention_mask is not None:
				attention_mask = attention_mask[:, -self.max_length :]
		return {"input_ids": input_ids, "attention_mask": attention_mask}

	def _encode_one_keep_suffix_with_offsets(self, text: str) -> dict:
		"""编码单条文本，并在超长时保留末尾 suffix，同时返回 offset_mapping。

		注意：
		- 依赖 fast tokenizer 才能返回 offset_mapping。
		- offset_mapping 的坐标是相对于原始字符串的字符位置；
		  当我们切片保留最后 max_length 个 token 时，一并切片 offsets 即可。
		"""
		try:
			enc = self.tokenizer(
				text,
				return_tensors="pt",
				add_special_tokens=True,
				truncation=False,
				return_offsets_mapping=True,
			)
		except Exception:
			# Slow tokenizers may not support return_offsets_mapping. Fall back to ids-only encode.
			enc_simple = self._encode_one_keep_suffix(text)
			enc_simple["offset_mapping"] = None
			return enc_simple
		input_ids: torch.Tensor = enc["input_ids"].to(self.device)
		attention_mask: Optional[torch.Tensor] = enc.get("attention_mask")
		if attention_mask is not None:
			attention_mask = attention_mask.to(self.device)
		offset_mapping = enc.get("offset_mapping")
		if offset_mapping is None:
			# Some tokenizers may silently drop offsets; behave like slow-tokenizer fallback.
			enc_simple = self._encode_one_keep_suffix(text)
			enc_simple["offset_mapping"] = None
			return enc_simple
		offset_mapping = offset_mapping.to(self.device)

		if input_ids.shape[1] > self.max_length:
			input_ids = input_ids[:, -self.max_length :]
			offset_mapping = offset_mapping[:, -self.max_length :, :]
			if attention_mask is not None:
				attention_mask = attention_mask[:, -self.max_length :]
		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"offset_mapping": offset_mapping,
		}

	@staticmethod
	def _strip_score_prefix(prompt: str) -> str:
		# 旧版逻辑：剥离 Score: [ 后缀。现已废弃，直接使用原始 prompt。
		return prompt

	def _extract_features(self, inputs: Sequence[Any]) -> torch.Tensor:
		"""抽取每条文本的向量特征，shape=(B, H)。

		做法：取最后一层 hidden state 中“最后一个非 padding token”的向量。
		"""

		batch = self._encode(inputs)
		hidden_states = self._forward_last_hidden_state(batch)
		attention_mask: torch.Tensor = batch.get("attention_mask")
		return self._pool_last_token_features(hidden_states, attention_mask)

	def predict_proba(self, inputs: Sequence[Any]) -> np.ndarray:
		inference_batch_size = 4
		all_results = []

		for start_idx in range(0, len(inputs), inference_batch_size):
			batch_inputs = inputs[start_idx : start_idx + inference_batch_size]

			if self.predict_mode == "lm_head_scores":
				assert self._score_token_map is not None
				# 修改：使用 generate(output_scores=True) 一次性获取 reasoning 和 scores
				prompts: List[str] = []
				for x in batch_inputs:
					prompts.append(str(x))

				self.model.eval()
				
				# 根据 fix_score_prefix_in_prompt / anchor_score_prefix 决定处理逻辑
				if self.fix_score_prefix_in_prompt:
					# 方案 A: prompt 已包含 "Score: ["，直接生成分数
					batch_probs = self._predict_with_fixed_prefix(prompts)
				else:
					if self.anchor_score_prefix:
						# 方案 C（推荐兜底）: prompt 不包含 "Score: ["，但我们在推理时临时锚定
						anchored_prompts = [p.rstrip() + "\nScore: [" for p in prompts]
						batch_probs = self._predict_with_fixed_prefix(anchored_prompts)
					else:
						# 方案 B: prompt 不包含 "Score: ["，需要在生成中查找
						batch_probs = self._predict_with_search(prompts)

				all_results.append(np.array(batch_probs, dtype=np.float32))
			elif self.predict_mode == "lm_head_multidim_scores":
				assert self._score_token_map is not None
				batch_probs = self._predict_multidim_scores(batch_inputs)
				all_results.append(np.array(batch_probs, dtype=np.float32))
			else:
				self.model.eval()
				assert self.classifier is not None
				self.classifier.eval()
				with torch.no_grad():
					features = self._extract_features(batch_inputs)
					logits = self.classifier(features)
					probs = torch.softmax(logits, dim=-1)
				all_results.append(probs.detach().cpu().numpy().astype(np.float32))

		if not all_results:
			return np.array([])
		
		# 强制触发一次垃圾回收，释放 generate 产生的临时大张量
		import gc
		gc.collect()
		torch.cuda.empty_cache()
		
		# For multidim mode, all_results elements are (B, D, K)
		return np.concatenate(all_results, axis=0)

	@staticmethod
	def _find_subsequence_last(seq: List[int], sub: List[int]) -> int:
		"""Return the last start index where sub occurs in seq, else -1."""
		if not sub or not seq or len(sub) > len(seq):
			return -1
		last = -1
		for i in range(0, len(seq) - len(sub) + 1):
			if seq[i : i + len(sub)] == sub:
				last = i
		return last

	def _predict_multidim_scores(self, inputs: Sequence[Any]) -> List[np.ndarray]:
		"""Predict per-dimension score distributions.

		Returns a list of np.ndarray, each shape=(D, K).
		"""
		import numpy as _np

		if not self.multidim_dimensions:
			raise ValueError("multidim_dimensions not set")
		assert self._score_token_map is not None
		assert hasattr(self, "_md_template_open_ids")
		assert hasattr(self, "_md_bracket_pos_in_template")

		# score -> first token id
		score_target_ids: Dict[int, int] = {}
		for score_val in self._score_values:
			info = self._score_token_map[score_val]
			score_target_ids[score_val] = info["ids"][0] if info.get("ids") else -1

		self.model.eval()
		out: List[_np.ndarray] = []
		with torch.no_grad():
			for x in inputs:
				prompt = str(x)
				enc = self._encode_one_keep_suffix(prompt)
				input_ids = enc["input_ids"]  # (1, L)
				attn = enc.get("attention_mask")
				outputs = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
				logits: torch.Tensor = outputs.logits  # (1, L, V)

				ids_list = input_ids[0].tolist()
				tpl = list(getattr(self, "_md_template_open_ids", []))
				if not tpl or len(ids_list) < len(tpl):
					# fallback: uniform
					out.append(_np.ones((len(self.multidim_dimensions), self.num_labels), dtype=_np.float32) / float(self.num_labels))
					continue
				# assume template is at the end
				tpl_start = len(ids_list) - len(tpl)
				# verify quick match; if mismatch, try to search last occurrence
				if ids_list[tpl_start:] != tpl:
					found = self._find_subsequence_last(ids_list, tpl)
					if found >= 0:
						tpl_start = found
					else:
						out.append(_np.ones((len(self.multidim_dimensions), self.num_labels), dtype=_np.float32) / float(self.num_labels))
						continue

				dim_probs: List[_np.ndarray] = []
				for bpos in getattr(self, "_md_bracket_pos_in_template"):
					pos = int(tpl_start + int(bpos))
					# logits at position pos predicts token pos+1 (score)
					step_logits = logits[0, pos, :]
					logit_vals = []
					for score_val in self._score_values:
						tid = score_target_ids[score_val]
						logit_vals.append(step_logits[tid].item() if tid != -1 else -1e9)
					arr = _np.array(logit_vals, dtype=_np.float64)
					arr -= _np.max(arr)
					exps = _np.exp(arr)
					probs = (exps / (_np.sum(exps) + 1e-12)).astype(_np.float32)
					dim_probs.append(probs)
				out.append(_np.stack(dim_probs, axis=0))

				del outputs, logits
		return out

	def _predict_with_fixed_prefix(self, prompts: List[str]) -> List[np.ndarray]:
		"""方案 A: prompt 已包含 "Score: ["，直接生成分数。
		
		这种情况下，生成的第一个 token 就应该是分数数字。
		我们只需要生成 1-2 个 token（单数字或 "10"），然后提取第一个 token 的 logits。
		
		返回：
		    List[np.ndarray]: 每个元素是 shape=(num_labels,) 的概率数组，
		                      索引 0..9 对应 label 0..9（即 score 1..10）
		"""
		batch_probs = []
		
		# 预先获取分数对应的 token IDs (取第一个 token)
		# 注意：_score_values 是 [1, 2, ..., 10]，对应 label [0, 1, ..., 9]
		score_target_ids = {}
		for score_val in self._score_values:
			info = self._score_token_map[score_val]
			score_target_ids[score_val] = info['ids'][0] if info['ids'] else -1
		
		# 1. 批量编码
		enc = self.tokenizer(
			prompts, 
			padding=True, 
			truncation=True,
			max_length=self.max_length, 
			return_tensors="pt"
		)
		input_ids = enc.input_ids.to(self.device)
		attention_mask = enc.attention_mask.to(self.device)

		# 2. 生成 (只需要生成 1-2 个 token)
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=min(3, int(self.max_new_tokens)),  # 只需要生成分数，最多 3 个 token
				do_sample=False,
				temperature=None,
				top_p=None,
				pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
				output_scores=True,
				return_dict_in_generate=True,
			)
		
		# 3. 提取第一个生成步的 logits（即分数的 logits）
		if not outputs.scores or len(outputs.scores) == 0:
			# 没有生成任何 token，返回均匀分布
			for _ in range(len(prompts)):
				batch_probs.append(np.ones(self.num_labels, dtype=np.float32) / self.num_labels)
		else:
			# outputs.scores[0] 是第一个生成步的 logits，shape=(Batch, Vocab)
			first_step_logits = outputs.scores[0]  # (Batch, Vocab)
			
			for i in range(len(prompts)):
				step_logits = first_step_logits[i]  # (Vocab,)
				
				# 提取每个 score 对应的 logit
				# _score_values 是按顺序的 [score_min, ..., score_max]
				# 返回的概率数组索引对应 label = score - score_min
				logit_vals = []
				for score_val in self._score_values:
					tid = score_target_ids[score_val]
					if tid != -1:
						logit_vals.append(step_logits[tid].item())
					else:
						logit_vals.append(-1e9)
				
				# Softmax
				inputs_np = np.array(logit_vals, dtype=np.float64)
				inputs_np -= np.max(inputs_np)  # Numerical stability
				exps = np.exp(inputs_np)
				probs = exps / (np.sum(exps) + 1e-12)
				batch_probs.append(probs.astype(np.float32))
		
		del outputs
		return batch_probs

	def _predict_with_search(self, prompts: List[str]) -> List[np.ndarray]:
		"""方案 B: prompt 不包含 "Score: ["，需要在生成中查找。
		
		这是原来的逻辑，需要生成更多 token，然后在生成的文本中搜索 "Score: [" 的位置。
		
		返回：
		    List[np.ndarray]: 每个元素是 shape=(num_labels,) 的概率数组，
		                      索引 0..9 对应 label 0..9（即 score 1..10）
		"""
		batch_probs = []
		
		# 预先获取分数对应的 token IDs (取第一个 token)
		# 注意：_score_values 是 [1, 2, ..., 10]，对应 label [0, 1, ..., 9]
		score_target_ids = {}
		for score_val in self._score_values:
			info = self._score_token_map[score_val]
			score_target_ids[score_val] = info['ids'][0] if info['ids'] else -1
		
		# 1. 批量编码
		enc = self.tokenizer(
			prompts, 
			padding=True, 
			truncation=True,
			max_length=self.max_length, 
			return_tensors="pt"
		)
		input_ids = enc.input_ids.to(self.device)
		attention_mask = enc.attention_mask.to(self.device)

		# 2. 生成 (带 scores)
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=int(self.max_new_tokens),
				do_sample=False,
				temperature=None,
				top_p=None,
				pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
				output_scores=True,
				return_dict_in_generate=True,
			)
		
		# 3. 解析 Logits
		generated_sequences = outputs.sequences[:, input_ids.shape[1]:]  # (Batch, GenLen)
		
		fallback_anchored: List[str] = []
		fallback_indices: List[int] = []
		for i in range(len(prompts)):
			# 当前样本生成的 token 序列
			gen_seq = generated_sequences[i]
			
			# 寻找 `Score: [` 的位置
			full_text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
			if "Score: [" not in full_text:
				# 没生成分数：
				# - 若 ignore_unparsable_score=True：返回 NaN 让上层忽略
				# - 否则：若开启 anchor 则兜底；否则返回均匀分布
				if self.ignore_unparsable_score:
					batch_probs.append(np.full(self.num_labels, np.nan, dtype=np.float32))
				elif self.anchor_score_prefix:
					fallback_anchored.append(prompts[i].rstrip() + "\nScore: [")
					fallback_indices.append(i)
					batch_probs.append(np.ones(self.num_labels, dtype=np.float32) / self.num_labels)
				else:
					batch_probs.append(np.ones(self.num_labels, dtype=np.float32) / self.num_labels)
				continue

			# 在 generated tokens 中寻找 `[` 对应的位置
			found_idx = -1
			gen_ids = gen_seq.tolist()
			
			# 从后往前找 '[' 的 token
			for t_idx in range(len(gen_ids) - 1, -1, -1):
				tok_str = self.tokenizer.decode([gen_ids[t_idx]])
				if "[" in tok_str and "Score" in self.tokenizer.decode(gen_ids[max(0, t_idx-5):t_idx+1]):
					# 找到了 Score: [ 中的 [
					found_idx = t_idx
					break
			
			if found_idx != -1 and found_idx + 1 < len(outputs.scores):
				# found_idx 是 '[' 的位置
				# 我们需要 '[' 后面那个 token (即分数) 对应的 logits
				# outputs.scores[k] 是生成第 k 个 token 时的 logits
				# 比如生成的 token 序列是 [Score, :, [, 5]
				# indices: 0, 1, 2, 3
				# '[' 在 index 2. 我们要 index 3 (分数 5) 的 logits.
				# 所以取 scores[found_idx + 1]
				step_logits = outputs.scores[found_idx + 1][i]  # (Vocab,)
				
				# 提取每个 score 对应的 logit
				# _score_values 是按顺序的 [score_min, ..., score_max]
				# 返回的概率数组索引对应 label = score - score_min
				logit_vals = []
				for score_val in self._score_values:
					tid = score_target_ids[score_val]
					if tid != -1:
						logit_vals.append(step_logits[tid].item())
					else:
						logit_vals.append(-1e9)
				
				# Softmax
				inputs_np = np.array(logit_vals, dtype=np.float64)
				inputs_np -= np.max(inputs_np)  # Numerical stability
				exps = np.exp(inputs_np)
				probs = exps / (np.sum(exps) + 1e-12)
				batch_probs.append(probs.astype(np.float32))
			else:
				# 没找到或越界：
				if self.ignore_unparsable_score:
					batch_probs.append(np.full(self.num_labels, np.nan, dtype=np.float32))
				elif self.anchor_score_prefix:
					fallback_anchored.append(prompts[i].rstrip() + "\nScore: [")
					fallback_indices.append(i)
					batch_probs.append(np.ones(self.num_labels, dtype=np.float32) / self.num_labels)
				else:
					batch_probs.append(np.ones(self.num_labels, dtype=np.float32) / self.num_labels)
		
		del outputs
		# 对失败样本做一次锚定兜底，避免全均匀分布导致 KL/选择器信号失真
		if fallback_anchored and (not self.ignore_unparsable_score):
			fallback_probs = self._predict_with_fixed_prefix(fallback_anchored)
			for pos, orig_i in enumerate(fallback_indices):
				batch_probs[orig_i] = fallback_probs[pos]
		return batch_probs

	def train_on_batch(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
		"""在一个 batch 的新标注上更新代理模型。

		- finetune_mode='lora'：参数高效微调（LoRA adapter + 可选分类头，显存占用显著更低）
		"""

		if self.predict_mode == "lm_head_scores":
			# 如果 prompt 不固定到 `Score: [`，并且数据里带有 reason，
			# 则做 CoT SFT：监督生成 `reason + \nScore: [X]`。
			# 否则保持原行为：只在分数 token 上做 teacher-forcing。
			if (not self.fix_score_prefix_in_prompt) and any(
				bool(str(getattr(x, "reason", "") or "").strip()) for x in inputs
			):
				self._train_on_batch_reason_then_score(inputs, labels)
			elif self.pointwise_loss_type == "coral":
				raise RuntimeError(
					"pointwise_loss_type='coral' requires LlamaSharedMultiTaskProxyModel, "
					"which adds a threshold head on top of hidden states."
				)
			elif self.pointwise_loss_type in {"ce_mse", "ce_cost"}:
				self._train_on_batch_score_tokens_ce_distance(inputs, labels)
			elif self.pointwise_loss_type == "ordinal":
				self._train_on_batch_score_tokens_ordinal(inputs, labels)
			else:
				self._train_on_batch_score_tokens(inputs, labels)
			return

		if self.predict_mode == "lm_head_multidim_scores":
			self._train_on_batch_multidim_score_tokens(inputs, labels)
			return

		assert self.classifier is not None
		self.classifier.train()
		self.model.train()

		# 使用 mini-batch 训练防止 OOM
		# 80G 显卡通常可以承受更大的 batch，但为了稳健这里默认设置为 8
		# 随着 active learning 进行，inputs 长度会增加到 1000+，一次性 encode 会导致 OOM
		mini_batch_size = 8
		n_samples = len(inputs)
		indices = np.arange(n_samples)
		np.random.shuffle(indices)

		for start_idx in range(0, n_samples, mini_batch_size):
			end_idx = min(start_idx + mini_batch_size, n_samples)
			batch_indices = indices[start_idx:end_idx]
			
			sub_inputs = [inputs[i] for i in batch_indices]
			sub_labels = [labels[i] for i in batch_indices]

			batch = self._encode(sub_inputs)
			labels_tensor = torch.tensor(sub_labels, dtype=torch.long, device=self.device)
			self.optimizer.zero_grad()
			attention_mask = batch.get("attention_mask")

			ac = self._autocast_ctx()
			if ac is not None:
				with ac:
					hidden_states = self._forward_last_hidden_state(batch)
					features = self._pool_last_token_features(hidden_states, attention_mask)
					logits = self.classifier(features)
					loss = self.loss_fn(logits, labels_tensor)
			else:
				hidden_states = self._forward_last_hidden_state(batch)
				features = self._pool_last_token_features(hidden_states, attention_mask)
				logits = self.classifier(features)
				loss = self.loss_fn(logits, labels_tensor)

			if self._scaler is not None:
				self._scaler.scale(loss).backward()
				self._scaler.step(self.optimizer)
				self._scaler.update()
			else:
				loss.backward()
				self.optimizer.step()

			# 下一个 mini-batch

			# 清理缓存
				del batch, labels_tensor, hidden_states, features, logits, loss

	def _build_filled_multidim_template_ids_and_score_positions(self, scores: Sequence[int]) -> tuple[List[int], List[int]]:
		"""Build filled template token ids and the positions of score tokens within it."""
		if not self.multidim_dimensions:
			raise ValueError("multidim_dimensions not set")
		if len(scores) != len(self.multidim_dimensions):
			raise ValueError("scores length mismatch with multidim_dimensions")
		ids_all: List[int] = []
		pos_score_tokens: List[int] = []
		cur = 0
		for i, (d, s) in enumerate(zip(self.multidim_dimensions, scores)):
			prefix = ("" if i == 0 else "\n") + f"{d}: ["
			score_text = str(int(s))
			suffix = "]"
			prefix_ids = self.tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
			score_ids = self.tokenizer(score_text, add_special_tokens=False).get("input_ids", [])
			suffix_ids = self.tokenizer(suffix, add_special_tokens=False).get("input_ids", [])
			ids_all.extend(prefix_ids)
			cur += len(prefix_ids)
			# score tokens positions
			for j in range(len(score_ids)):
				pos_score_tokens.append(cur + j)
			ids_all.extend(score_ids)
			cur += len(score_ids)
			ids_all.extend(suffix_ids)
			cur += len(suffix_ids)
		return ids_all, pos_score_tokens

	def _train_on_batch_multidim_score_tokens(self, inputs: Sequence[Any], labels: Sequence[Any]) -> None:
		"""Train only on the score tokens for each dimension (teacher-forcing).

		labels is expected to be shape (B, D) with values 0..(num_labels-1).
		"""
		self.model.train()
		device = self.device
		if not self.multidim_dimensions:
			raise ValueError("multidim_dimensions not set")
		d = len(self.multidim_dimensions)
		# normalize labels to list[list[int]]
		labels_list: List[List[int]] = []
		for y in labels:
			arr = np.asarray(y, dtype=np.int64)
			if arr.ndim != 1 or int(arr.size) != int(d):
				raise ValueError("multidim labels must be 1D vectors with length == num_dimensions")
			labels_list.append([int(v) for v in arr.tolist()])

		accumulation_steps = 4
		self.optimizer.zero_grad()
		step_count = 0
		aligned_samples = 0
		total_samples = 0

		for x, yvec in zip(inputs, labels_list):
			total_samples += 1
			scores = [int(v) + int(self.score_min) for v in yvec]
			prompt_prefix = getattr(x, "prompt_prefix", None)
			if prompt_prefix is None:
				# fallback: use full prompt string and hope template at end
				prompt_prefix = str(x).rsplit("\n", maxsplit=len(self.multidim_dimensions))
				prompt_prefix = prompt_prefix[0] if isinstance(prompt_prefix, list) else str(x)
			prefix = str(prompt_prefix).rstrip()
			template_lines = [f"{dim}: [{sc}]" for dim, sc in zip(self.multidim_dimensions, scores)]
			template_text = "\n".join(template_lines)
			train_text = prefix + "\n" + template_text

			enc_full = self._encode_one_keep_suffix_with_offsets(train_text)
			input_ids: torch.Tensor = enc_full["input_ids"]  # (1, L)
			attn = enc_full.get("attention_mask")
			offsets = enc_full.get("offset_mapping")  # (1, L, 2) or None
			ids_len = int(input_ids.shape[1])
			if ids_len <= 0:
				continue

			labels_ids = torch.full_like(input_ids, fill_value=-100)
			n_supervised = 0
			if offsets is not None:
				# Compute character spans for the score digits in the (filled) template region.
				# We supervise only these digit tokens; everything else is masked with -100.
				template_start_char = len(prefix) + 1  # the '\n' between prefix and template
				spans: List[tuple[int, int]] = []
				cur = 0
				for dim, sc, line in zip(self.multidim_dimensions, scores, template_lines):
					sc_str = str(int(sc))
					digit_off = len(f"{dim}: [")
					s = template_start_char + cur + digit_off
					e = s + len(sc_str)
					spans.append((int(s), int(e)))
					cur += len(line) + 1  # + '\n'

				# offsets are (start,end) char indices into the original string.
				# Special tokens may have (0,0); ignore them.
				for tok_i in range(ids_len):
					st = int(offsets[0, tok_i, 0].item())
					en = int(offsets[0, tok_i, 1].item())
					if st == 0 and en == 0:
						continue
					for (s, e) in spans:
						# overlap check
						if st < e and en > s:
							labels_ids[0, tok_i] = input_ids[0, tok_i]
							n_supervised += 1
							break
			else:
				# Fallback: offsets not available, use legacy token-subsequence alignment.
				# This is less reliable for unigram tokenizers, but avoids crashing.
				tpl_ids, tpl_score_pos = self._build_filled_multidim_template_ids_and_score_positions(scores)
				ids_list = input_ids[0].tolist()
				if len(ids_list) >= len(tpl_ids):
					tpl_start = len(ids_list) - len(tpl_ids)
					if ids_list[tpl_start:] != tpl_ids:
						found = self._find_subsequence_last(ids_list, tpl_ids)
						if found >= 0:
							tpl_start = found
						else:
							tpl_start = -1
					if tpl_start >= 0:
						for pos_in_tpl in tpl_score_pos:
							pos = int(tpl_start + int(pos_in_tpl))
							if 0 <= pos < ids_len:
								labels_ids[0, pos] = input_ids[0, pos]
								n_supervised += 1

			if n_supervised <= 0:
				continue
			aligned_samples += 1

			ac = self._autocast_ctx()
			if ac is not None:
				with ac:
					outputs = self.model(input_ids=input_ids, attention_mask=attn, labels=labels_ids)
					loss = outputs.loss
			else:
				outputs = self.model(input_ids=input_ids, attention_mask=attn, labels=labels_ids)
				loss = outputs.loss

			loss = loss / accumulation_steps
			if self._scaler is not None:
				self._scaler.scale(loss).backward()
			else:
				loss.backward()

			step_count += 1
			if step_count % accumulation_steps == 0:
				if self._scaler is not None:
					self._scaler.step(self.optimizer)
					self._scaler.update()
				else:
					self.optimizer.step()
				self.optimizer.zero_grad()
				del outputs, loss

		if step_count % accumulation_steps != 0:
			if self._scaler is not None:
				self._scaler.step(self.optimizer)
				self._scaler.update()
			else:
				self.optimizer.step()
			self.optimizer.zero_grad()

		# If we never supervised any token, training is a no-op; warn once to make it obvious.
		if step_count <= 0 and total_samples > 0:
			if not bool(getattr(self, "_warned_multidim_no_steps", False)):
				print(
					"WARNING: multidim train_on_batch supervised 0 samples (all alignment skipped). "
					"Proxy will not update; check prompt/template formatting and max_length truncation.",
					flush=True,
				)
				setattr(self, "_warned_multidim_no_steps", True)

		# Optional lightweight debug stats
		if bool(getattr(self, "debug_multidim_alignment", False)):
			print(
				f"[multidim-train] aligned_samples={aligned_samples}/{total_samples} "
				f"optimizer_steps={(step_count // accumulation_steps) + (1 if (step_count % accumulation_steps) else 0)}",
				flush=True,
			)



	def _train_on_batch_score_tokens(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
		"""lm_head_scores 模式的训练：只在分数 token 上做 teacher-forcing。

		要求：
		- labels 是 0..9（对应 score_min..score_max）
		- 如果 fix_score_prefix_in_prompt=True: prompt 文本结尾已固定到 `Score: [`，
		  这样第一个要预测的 token 就是数字。
		- 如果 fix_score_prefix_in_prompt=False: 需要在 prompt 后拼接 `Score: [` 再训练。
		"""
		import torch.nn.functional as F

		self.model.train()
		device = self.device
		
		# 使用梯度累积防止 OOM
		accumulation_steps = 8
		self.optimizer.zero_grad()
		step_count = 0

		for x, y in zip(inputs, labels):
			score = int(y) + int(self.score_min)
			prompt = str(x)
			sample_weight = 1.0
			if self.pointwise_class_weights is not None:
				label_idx = int(y)
				if 0 <= label_idx < int(self.pointwise_class_weights.numel()):
					sample_weight = float(self.pointwise_class_weights[label_idx].item())
			
			# 根据配置决定是否需要手动添加 "Score: ["
			if not self.fix_score_prefix_in_prompt:
				# prompt 不包含 "Score: ["，需要手动添加
				if not prompt.endswith("Score: ["):
					prompt = prompt.rstrip() + "\nScore: ["
			
			# prompt（超长时保留末尾，确保 `Score: [` 仍在上下文里）
			enc_prompt = self._encode_one_keep_suffix(prompt)
			prompt_ids = enc_prompt["input_ids"][0]
			# score tokens (digits only)
			score_ids_list = self.tokenizer(str(score), add_special_tokens=False).get("input_ids", [])
			if not score_ids_list:
				continue
			score_ids = torch.tensor(score_ids_list, dtype=torch.long, device=device)

			input_ids = torch.cat([prompt_ids, score_ids], dim=0).unsqueeze(0)  # (1, L)
			attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

			loss = torch.zeros((), device=device)

			ac = self._autocast_ctx()
			if ac is not None:
				with ac:
					outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
					logits = outputs.logits  # (1, L, V)
					prompt_len = int(prompt_ids.numel())
					for i, tid in enumerate(score_ids_list):
						pos = prompt_len + i - 1
						loss = loss + F.cross_entropy(logits[0, pos, :].unsqueeze(0), torch.tensor([tid], device=device))
			else:
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				logits = outputs.logits
				prompt_len = int(prompt_ids.numel())
				for i, tid in enumerate(score_ids_list):
					pos = prompt_len + i - 1
					loss = loss + F.cross_entropy(logits[0, pos, :].unsqueeze(0), torch.tensor([tid], device=device))

			loss = loss * float(sample_weight)

			# Scale loss
			loss = loss / accumulation_steps
			if self._scaler is not None:
				self._scaler.scale(loss).backward()
			else:
				loss.backward()

			step_count += 1
			if step_count % accumulation_steps == 0:
				if self._scaler is not None:
					self._scaler.step(self.optimizer)
					self._scaler.update()
				else:
					self.optimizer.step()
				self.optimizer.zero_grad()
				# 及时清理
				del outputs, logits, loss, input_ids, attention_mask
		
		# 处理剩余的梯度
		if step_count % accumulation_steps != 0:
			if self._scaler is not None:
				self._scaler.step(self.optimizer)
				self._scaler.update()
			else:
				self.optimizer.step()
			self.optimizer.zero_grad()

	def _train_on_batch_score_tokens_ce_distance(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
		self.model.train()
		device = self.device

		accumulation_steps = 8
		self.optimizer.zero_grad()
		step_count = 0

		for x, y in zip(inputs, labels):
			label_idx = int(y)
			score = label_idx + int(self.score_min)
			prompt = str(x)
			sample_weight = 1.0
			if self.pointwise_class_weights is not None:
				if 0 <= label_idx < int(self.pointwise_class_weights.numel()):
					sample_weight = float(self.pointwise_class_weights[label_idx].item())

			if not self.fix_score_prefix_in_prompt and not prompt.endswith("Score: ["):
				prompt = prompt.rstrip() + "\nScore: ["

			enc_prompt = self._encode_one_keep_suffix(prompt)
			prompt_ids = enc_prompt["input_ids"][0]
			score_ids_list = self.tokenizer(str(score), add_special_tokens=False).get("input_ids", [])
			if not score_ids_list:
				continue
			score_ids = torch.tensor(score_ids_list, dtype=torch.long, device=device)

			input_ids = torch.cat([prompt_ids, score_ids], dim=0).unsqueeze(0)
			attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
			prompt_len = int(prompt_ids.numel())
			loss = torch.zeros((), device=device)

			ac = self._autocast_ctx()
			if ac is not None:
				with ac:
					outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
					logits = outputs.logits
			else:
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				logits = outputs.logits

			for i, tid in enumerate(score_ids_list):
				pos = int(prompt_len + i - 1)
				step_logits = logits[0, pos, :]
				if i == 0:
					score_logits = self._score_logits_from_step_logits(step_logits)
					loss = loss + self._ce_distance_loss_from_score_logits(
						score_logits,
						target_label=label_idx,
						target_token_id=int(tid),
						full_vocab_logits=step_logits,
					)
				else:
					import torch.nn.functional as F

					loss = loss + F.cross_entropy(
						step_logits.unsqueeze(0),
						torch.tensor([int(tid)], dtype=torch.long, device=device),
					)

			loss = loss * float(sample_weight)
			loss = loss / accumulation_steps
			if self._scaler is not None:
				self._scaler.scale(loss).backward()
			else:
				loss.backward()

			step_count += 1
			if step_count % accumulation_steps == 0:
				if self._scaler is not None:
					self._scaler.step(self.optimizer)
					self._scaler.update()
				else:
					self.optimizer.step()
				self.optimizer.zero_grad()
				del outputs, logits, loss, input_ids, attention_mask

		if step_count % accumulation_steps != 0:
			if self._scaler is not None:
				self._scaler.step(self.optimizer)
				self._scaler.update()
			else:
				self.optimizer.step()
			self.optimizer.zero_grad()

	def _score_token_first_ids(self) -> Dict[int, int]:
		if self._score_token_map is None or self._score_values is None:
			raise RuntimeError("score token map is not initialized")
		score_target_ids: Dict[int, int] = {}
		for score_val in self._score_values:
			info = self._score_token_map[int(score_val)]
			ids = info.get("ids", [])
			score_target_ids[int(score_val)] = int(ids[0]) if ids else -1
		return score_target_ids

	def _score_logits_from_step_logits(self, step_logits: torch.Tensor) -> torch.Tensor:
		score_target_ids = self._score_token_first_ids()
		logit_vals: List[torch.Tensor] = []
		for score_val in self._score_values or []:
			tid = int(score_target_ids[int(score_val)])
			if tid >= 0:
				logit_vals.append(step_logits[int(tid)])
			else:
				logit_vals.append(torch.full((), -1e9, dtype=step_logits.dtype, device=step_logits.device))
		if not logit_vals:
			raise RuntimeError("no score token ids available for ordinal pointwise loss")
		return torch.stack(logit_vals, dim=0)

	def _ce_distance_loss_from_score_logits(
		self,
		score_logits: torch.Tensor,
		*,
		target_label: int,
		target_token_id: int,
		full_vocab_logits: torch.Tensor,
	) -> torch.Tensor:
		import torch.nn.functional as F

		device = score_logits.device
		ce_loss = F.cross_entropy(
			full_vocab_logits.unsqueeze(0),
			torch.tensor([int(target_token_id)], dtype=torch.long, device=device),
		)
		weight = float(self.pointwise_distance_weight)
		if weight <= 0.0:
			return ce_loss

		score_probs = torch.softmax(score_logits.float(), dim=-1)
		score_values = torch.arange(
			int(self.score_min),
			int(self.score_max) + 1,
			dtype=torch.float32,
			device=device,
		)
		target_score = torch.tensor(
			float(int(target_label) + int(self.score_min)),
			dtype=torch.float32,
			device=device,
		)
		scale = max(1.0, float(int(self.score_max) - int(self.score_min)))
		if self.pointwise_loss_type == "ce_cost":
			cost = torch.abs(score_values - target_score) / scale
			distance_loss = torch.sum(score_probs * cost)
		else:
			pred_score = torch.sum(score_probs * score_values)
			distance_loss = ((pred_score - target_score) / scale).pow(2)
		return ce_loss + weight * distance_loss

	def _ordinal_threshold_probs_from_score_logits(self, score_logits: torch.Tensor) -> torch.Tensor:
		score_probs = torch.softmax(score_logits, dim=-1)
		if int(score_probs.numel()) <= 1:
			return torch.empty((0,), dtype=score_probs.dtype, device=score_probs.device)
		return torch.flip(torch.cumsum(torch.flip(score_probs[1:], dims=(0,)), dim=0), dims=(0,))

	def _ordinal_score_probs_from_threshold_probs_np(self, threshold_probs: np.ndarray) -> np.ndarray:
		threshold_probs = np.asarray(threshold_probs, dtype=np.float64)
		if threshold_probs.size != max(0, int(self.num_labels) - 1):
			return np.ones(int(self.num_labels), dtype=np.float32) / float(self.num_labels)
		threshold_probs = np.clip(threshold_probs, 0.0, 1.0)
		# Enforce monotonic P(y>t) so differences form a valid score distribution.
		for i in range(1, threshold_probs.size):
			if threshold_probs[i] > threshold_probs[i - 1]:
				threshold_probs[i] = threshold_probs[i - 1]
		prev = np.concatenate(([1.0], threshold_probs))
		nxt = np.concatenate((threshold_probs, [0.0]))
		probs = np.maximum(prev - nxt, 0.0)
		total = float(probs.sum())
		if total <= 1e-12:
			return np.ones(int(self.num_labels), dtype=np.float32) / float(self.num_labels)
		return (probs / total).astype(np.float32)

	def _train_on_batch_score_tokens_ordinal(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
		"""Ordinal pointwise loss over score-token logits.

		The model still produces logits for score tokens 1..10, but training compares the
		cumulative probabilities P(score > threshold) against binary threshold labels.
		This keeps the existing LM-head score interface while giving the loss an ordered
		notion of distance between scores.
		"""
		import torch.nn.functional as F

		self.model.train()
		device = self.device
		accumulation_steps = 8
		self.optimizer.zero_grad()
		step_count = 0

		for x, y in zip(inputs, labels):
			label_idx = int(y)
			score = label_idx + int(self.score_min)
			if label_idx < 0 or label_idx >= int(self.num_labels):
				continue

			prompt = str(x)
			sample_weight = 1.0
			if self.pointwise_class_weights is not None:
				if 0 <= label_idx < int(self.pointwise_class_weights.numel()):
					sample_weight = float(self.pointwise_class_weights[label_idx].item())

			if not self.fix_score_prefix_in_prompt and not prompt.endswith("Score: ["):
				prompt = prompt.rstrip() + "\nScore: ["

			enc_prompt = self._encode_one_keep_suffix(prompt)
			prompt_ids = enc_prompt["input_ids"][0]
			score_ids_list = self.tokenizer(str(score), add_special_tokens=False).get("input_ids", [])
			if not score_ids_list:
				continue
			score_ids = torch.tensor(score_ids_list, dtype=torch.long, device=device)
			input_ids = torch.cat([prompt_ids, score_ids], dim=0).unsqueeze(0)
			attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
			prompt_len = int(prompt_ids.numel())
			pos = int(prompt_len - 1)
			targets = (
				torch.arange(0, int(self.num_labels) - 1, dtype=torch.float32, device=device)
				< float(label_idx)
			).to(dtype=torch.float32)

			ac = self._autocast_ctx()
			if ac is not None:
				with ac:
					outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
					logits = outputs.logits
					score_logits = self._score_logits_from_step_logits(logits[0, pos, :])
					threshold_probs = self._ordinal_threshold_probs_from_score_logits(score_logits)
			else:
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				logits = outputs.logits
				score_logits = self._score_logits_from_step_logits(logits[0, pos, :])
				threshold_probs = self._ordinal_threshold_probs_from_score_logits(score_logits)
			loss = F.binary_cross_entropy(
				threshold_probs.float().clamp(min=1e-6, max=1.0 - 1e-6),
				targets,
				reduction="mean",
			)

			loss = loss * float(sample_weight)
			loss = loss / accumulation_steps
			if self._scaler is not None:
				self._scaler.scale(loss).backward()
			else:
				loss.backward()

			step_count += 1
			if step_count % accumulation_steps == 0:
				if self._scaler is not None:
					self._scaler.step(self.optimizer)
					self._scaler.update()
				else:
					self.optimizer.step()
				self.optimizer.zero_grad()
				del outputs, logits, loss, input_ids, attention_mask

		if step_count % accumulation_steps != 0:
			if self._scaler is not None:
				self._scaler.step(self.optimizer)
				self._scaler.update()
			else:
				self.optimizer.step()
			self.optimizer.zero_grad()

	def _train_on_batch_reason_then_score(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
		"""lm_head_scores + no fixed prefix 下的 CoT 训练。

		目标：让模型先生成一段解释（reason），再生成 `Score: [X]`。
		做法：对 prompt 部分做 loss mask（labels=-100），只在目标输出上计算 LM loss。
		
		注意：这会显著比“只训分数 token”更慢；建议配合较小 batch/较少 epochs 使用。
		"""
		self.model.train()
		device = self.device

		# reasoning 较长时容易 OOM，这里比 score-only 更保守一些
		accumulation_steps = 4
		self.optimizer.zero_grad()
		step_count = 0

		for x, y in zip(inputs, labels):
			reason = str(getattr(x, "reason", "") or "").strip()
			if not reason:
				# 没有 reason 的样本退回 score-only
				continue
			score = int(y) + int(self.score_min)
			prompt = str(x)
			# 确保 prompt 不以固定前缀结尾（否则会导致先生成分数而不是 reason）
			if prompt.rstrip().endswith("Score: ["):
				prompt = prompt.rstrip()[: -len("Score: [")].rstrip()

			# 目标输出：reason + Score（可选截断 reason token，但保证 Score 后缀保留）
			reason_ids_list = self.tokenizer(reason.rstrip(), add_special_tokens=False).get("input_ids", [])
			if self.reason_max_tokens > 0 and len(reason_ids_list) > int(self.reason_max_tokens):
				reason_ids_list = reason_ids_list[: int(self.reason_max_tokens)]
			suffix_text = "\nScore: [" + str(score) + "]"
			suffix_ids_list = self.tokenizer(suffix_text, add_special_tokens=False).get("input_ids", [])
			target_ids_list = list(reason_ids_list) + list(suffix_ids_list)

			# prompt 编码：超长时保留末尾（保留 Judge 区域与候选输出）
			enc_prompt = self._encode_one_keep_suffix(prompt)
			prompt_ids = enc_prompt["input_ids"][0]
			if not target_ids_list:
				continue
			target_ids = torch.tensor(target_ids_list, dtype=torch.long, device=device)

			full_ids = torch.cat([prompt_ids, target_ids], dim=0)  # (L,)
			prompt_len = int(prompt_ids.numel())

			# 若超长：保留最后 max_length 个 token，同时修正 prompt_len
			if int(full_ids.numel()) > int(self.max_length):
				over = int(full_ids.numel()) - int(self.max_length)
				full_ids = full_ids[over:]
				prompt_len = max(0, prompt_len - over)

			input_ids = full_ids.unsqueeze(0)  # (1, L)
			attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
			labels_ids = full_ids.clone()
			if prompt_len > 0:
				labels_ids[:prompt_len] = -100
			labels_ids = labels_ids.unsqueeze(0)

			ac = self._autocast_ctx()
			if ac is not None:
				with ac:
					outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_ids)
					loss = outputs.loss
			else:
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_ids)
				loss = outputs.loss

			loss = loss / accumulation_steps
			if self._scaler is not None:
				self._scaler.scale(loss).backward()
			else:
				loss.backward()

			step_count += 1
			if step_count % accumulation_steps == 0:
				if self._scaler is not None:
					self._scaler.step(self.optimizer)
					self._scaler.update()
				else:
					self.optimizer.step()
				self.optimizer.zero_grad()
				del outputs, loss, input_ids, attention_mask, labels_ids

		# 处理剩余梯度
		if step_count % accumulation_steps != 0:
			if self._scaler is not None:
				self._scaler.step(self.optimizer)
				self._scaler.update()
			else:
				self.optimizer.step()
			self.optimizer.zero_grad()

	def clone(self) -> "LlamaProxyModel":
		"""轻量 clone：本实现不支持。

		LoRA/LLM 权重体积很大，复制代价高；上层应使用近似方法。
		"""
		raise NotImplementedError("LlamaProxyModel.clone() is not supported in LoRA-only mode.")

		new_model = object.__new__(LlamaProxyModel)
		new_model.model_path = self.model_path
		new_model.num_labels = self.num_labels
		new_model.hidden_size = self.hidden_size
		new_model.device = self.device
		new_model.max_length = self.max_length

		new_model.tokenizer = self.tokenizer
		new_model.model = self.model
		new_model.finetune_mode = self.finetune_mode
		new_model.use_amp = False
		new_model._scaler = None

		new_model.classifier = nn.Linear(self.hidden_size, self.num_labels).to(self.device)
		new_model.classifier.load_state_dict(self.classifier.state_dict())
		new_model.optimizer = torch.optim.AdamW(
			new_model.classifier.parameters(),
			lr=self.optimizer.param_groups[0]["lr"],
			weight_decay=self.optimizer.param_groups[0].get("weight_decay", 0.0),
		)
		new_model.loss_fn = nn.CrossEntropyLoss()
		return new_model

	def get_tokens_for_scores(self, scores: Sequence[int]) -> dict:
		"""返回给定分数字符串对应的 tokenizer ids 与 token 文本表示。

		参数
		------
		scores: Sequence[int] - 一组整数分数（例如 1..10）

		返回值
		------
		dict: 映射 score -> {"text": str, "ids": List[int], "tokens": List[str]}

		注意：Tokenizer 可能把某些数字拆成多个 token；返回的 "ids" 与 "tokens" 反映 tokenizer 的实际划分。
		"""
		mapping = {}
		for s in scores:
			text = str(int(s))
			# 不添加 special tokens，直接查看原始 tokenization
			enc = self.tokenizer(text, add_special_tokens=False)
			ids = enc.get("input_ids", [])
			# convert_ids_to_tokens 可能在不同 tokenizer 中命名不同，但大多数 transformers tokenizer 支持
			tokens = self.tokenizer.convert_ids_to_tokens(ids) if ids else []
			mapping[int(s)] = {"text": text, "ids": list(ids), "tokens": list(tokens)}
		return mapping

	def predict_proba_via_lm_head(self, inputs: Sequence[Any], score_token_map: dict, temperature: float = 1.0) -> np.ndarray:
		"""使用 LM 的 vocab logits 将每个分数映射为概率分布。

		参数
		------
		inputs: Sequence[Any] - 输入文本序列
		score_token_map: dict - score -> {"ids": List[int], ...}，例如由 `get_tokens_for_scores` 返回
		temperature: float - softmax 温度（>0），用于调节置信度

		返回
		------
		np.ndarray shape=(len(inputs), num_scores) 的概率矩阵，按 score_token_map 中按 key 升序的 scores 返回列顺序。

		说明
		----
		对于每个输入与每个 score 的 token id 序列，逐 token 计算其条件 log-prob（按因式分解），得到该 score 的 log-prob，再对所有 score 做 softmax 归一化得到概率分布。
		此方法在多 token 序列或大 vocab 下会较慢，但概念上直接且不需要额外微调。
		"""
		import math
		self.model.eval()
		device = self.device
		scores = sorted(list(score_token_map.keys()))
		n_scores = len(scores)
		results = []
		for x in inputs:
			text = str(x)
			enc = self._encode_one_keep_suffix(text)
			input_ids_base = enc["input_ids"][0].tolist()
			logps = []
			for s in scores:
				ids = score_token_map[s].get("ids", [])
				if not ids:
					logps.append(-1e9)
					continue
				cur_ids = list(input_ids_base)
				total_logp = 0.0
				for tid in ids:
					# 保持滑动窗口，确保末尾的 `Score: [` 永远还在上下文里
					if len(cur_ids) > self.max_length:
						cur_ids = cur_ids[-self.max_length :]
					input_tensor = torch.tensor([cur_ids], device=device)
					with torch.no_grad():
						outputs = self.model(input_ids=input_tensor)
						logits = outputs.logits  # (1, seq_len, vocab)
						last_logits = logits[0, -1, :]
						log_probs = torch.log_softmax(last_logits / float(temperature), dim=-1)
						lp = log_probs[tid].item()
					total_logp += float(lp)
					cur_ids.append(int(tid))
				logps.append(total_logp)
			# stable softmax
			arr = np.array(logps, dtype=np.float64)
			arr = arr - np.max(arr)
			expa = np.exp(arr)
			probs = expa / (expa.sum() + 1e-12)
			results.append(probs.astype(np.float32))
		return np.stack(results, axis=0)

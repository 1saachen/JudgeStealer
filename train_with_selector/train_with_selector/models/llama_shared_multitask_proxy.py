from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .llama_shared_proxy import LlamaSharedProxyModel


class LlamaSharedMultiTaskProxyModel(LlamaSharedProxyModel):
    """Shared-backbone multitask proxy (Pointwise + Pairwise).

    This class supports two modes:

    - ``multitask_mode='lm_head'`` (DEFAULT):
        - No extra classification heads.
        - Pointwise uses Llama LM head to produce a score distribution (same as
          ``predict_mode='lm_head_scores'`` in :class:`~train_with_selector.models.llama_proxy.LlamaProxyModel`).
        - Pairwise uses Llama LM head to score the three exact outputs ``[[1]]``, ``[[2]]``, ``[[3]]``.

    - ``multitask_mode='classifier_heads'`` (legacy):
        - Uses last-token hidden state feature extraction + linear heads.
        - Pointwise head: 10-way; Pairwise head: 3-way.
    """

    TASK_POINTWISE = "pointwise"
    TASK_PAIRWISE = "pairwise"

    def __init__(
        self,
        *,
        model_path: str,
        pointwise_num_labels: int = 10,
        pairwise_num_labels: int = 3,
        multitask_mode: str = "lm_head",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_length: int = 512,
        device: str | None = None,
        trust_remote_code: bool = False,
        finetune_mode: str = "lora",
        gradient_checkpointing: bool = True,
        use_amp: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
        load_in_4bit: bool = True,
        # For pointwise lm_head_scores mode.
        score_min: int = 1,
        score_max: int = 10,
        fix_score_prefix_in_prompt: bool = True,
        anchor_score_prefix: bool = False,
        pointwise_loss_type: str = "ce",
        pointwise_distance_weight: float = 0.0,
    ) -> None:
        self.multitask_mode = str(multitask_mode)
        if self.multitask_mode not in {"lm_head", "classifier_heads"}:
            raise ValueError(
                "multitask_mode must be one of {'lm_head','classifier_heads'}, got " + str(multitask_mode)
            )

        self.pointwise_num_labels = int(pointwise_num_labels)
        self.pairwise_num_labels = int(pairwise_num_labels)
        self.pointwise_loss_type = str(pointwise_loss_type)
        if self.pointwise_loss_type not in {"ce", "ce_mse", "ce_cost", "ordinal", "coral"}:
            raise ValueError(
                "pointwise_loss_type must be one of {'ce','ce_mse','ce_cost','ordinal','coral'}, got "
                + str(pointwise_loss_type)
            )
        self.pointwise_distance_weight = float(pointwise_distance_weight)
        if self.pointwise_distance_weight < 0.0:
            raise ValueError("pointwise_distance_weight must be >= 0")

        if self.multitask_mode == "lm_head":
            # Use LM head for pointwise scores.
            super().__init__(
                model_path=model_path,
                num_labels=int(pointwise_num_labels),
                lr=float(lr),
                weight_decay=float(weight_decay),
                max_length=int(max_length),
                device=device,
                predict_mode="lm_head_scores",
                trust_remote_code=bool(trust_remote_code),
                finetune_mode=str(finetune_mode),
                gradient_checkpointing=bool(gradient_checkpointing),
                use_amp=bool(use_amp),
                lora_r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                lora_target_modules=str(lora_target_modules),
                load_in_4bit=bool(load_in_4bit),
                score_min=int(score_min),
                score_max=int(score_max),
                fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
                anchor_score_prefix=bool(anchor_score_prefix),
                pointwise_loss_type=str(pointwise_loss_type),
                pointwise_distance_weight=float(pointwise_distance_weight),
            )

            # Pairwise: score exact outputs via LM head (no extra head params).
            if int(self.pairwise_num_labels) != 3:
                raise ValueError(
                    "lm_head multitask currently expects pairwise_num_labels==3 for [[1]]/[[2]]/[[3]]."
                )
            self._pairwise_choice_texts: List[str] = ["\n[[1]]", "\n[[2]]", "\n[[3]]"]
            self._pairwise_choice_ids: List[List[int]] = [
                self.tokenizer(t, add_special_tokens=False).get("input_ids", []) for t in self._pairwise_choice_texts
            ]
            if any(len(ids) == 0 for ids in self._pairwise_choice_ids):
                raise RuntimeError("Failed to tokenize pairwise choice strings for LM-head scoring.")

            if self.pointwise_loss_type == "coral":
                self.pointwise_head = nn.Linear(
                    int(self.hidden_dim),
                    int(self.pointwise_num_labels) - 1,
                ).to(self.device)
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                trainable_params.extend(list(self.pointwise_head.parameters()))
                try:
                    self.optimizer = torch.optim.AdamW(
                        trainable_params,
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                        foreach=False,
                    )
                except TypeError:
                    self.optimizer = torch.optim.AdamW(
                        trainable_params,
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                    )
            else:
                self.pointwise_head = None
            self.pairwise_head = None
        else:
            # Legacy behavior: classifier heads.
            super().__init__(
                model_path=model_path,
                num_labels=int(pointwise_num_labels),
                lr=float(lr),
                weight_decay=float(weight_decay),
                max_length=int(max_length),
                device=device,
                predict_mode="classifier",
                trust_remote_code=bool(trust_remote_code),
                finetune_mode=str(finetune_mode),
                gradient_checkpointing=bool(gradient_checkpointing),
                use_amp=bool(use_amp),
                lora_r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                lora_target_modules=str(lora_target_modules),
                load_in_4bit=bool(load_in_4bit),
                pointwise_loss_type=str(pointwise_loss_type),
                pointwise_distance_weight=float(pointwise_distance_weight),
            )

            if self.classifier is None:
                raise RuntimeError("Expected classifier head to be initialized in predict_mode='classifier'.")

            if self.pointwise_loss_type == "coral":
                self.pointwise_head = nn.Linear(
                    int(self.hidden_dim),
                    int(self.pointwise_num_labels) - 1,
                ).to(self.device)
            else:
                # Rename the inherited single head as the pointwise head.
                self.pointwise_head = self.classifier
            self.pairwise_head: nn.Module = nn.Linear(int(self.hidden_dim), int(self.pairwise_num_labels)).to(self.device)

            # Rebuild optimizer to include BOTH heads + trainable backbone params (LoRA adapters or full).
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            trainable_params.extend(list(self.pointwise_head.parameters()))
            trainable_params.extend(list(self.pairwise_head.parameters()))

            try:
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                    foreach=False,
                )
            except TypeError:
                self.optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))

            self.loss_fn = nn.CrossEntropyLoss()

    # ---------------------------------------------------------------------
    # Public API (task-specific)
    # ---------------------------------------------------------------------

    def predict_proba_pointwise(self, inputs: Sequence[Any]) -> np.ndarray:
        if self.pointwise_loss_type == "coral":
            if self.pointwise_head is None:
                raise RuntimeError("CORAL pointwise head is not initialized.")
            return self._predict_proba_with_coral_head(inputs, head=self.pointwise_head)
        if self.multitask_mode == "lm_head":
            return super().predict_proba(inputs)
        return self._predict_proba_with_head(inputs, head=self.pointwise_head)

    def predict_proba_pairwise(self, inputs: Sequence[Any]) -> np.ndarray:
        if self.multitask_mode == "lm_head":
            return self._predict_proba_pairwise_lm_head(inputs)
        return self._predict_proba_with_head(inputs, head=self.pairwise_head)

    def train_on_batch_pointwise(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
        if self.pointwise_loss_type == "coral":
            if self.pointwise_head is None:
                raise RuntimeError("CORAL pointwise head is not initialized.")
            self._train_on_batch_with_coral_head(inputs, labels, head=self.pointwise_head)
            return
        if self.multitask_mode == "lm_head":
            super().train_on_batch(inputs, labels)
            return
        if self.pointwise_loss_type == "ordinal":
            self._train_on_batch_with_head_ordinal(inputs, labels, head=self.pointwise_head)
            return
        self._train_on_batch_with_head(inputs, labels, head=self.pointwise_head)

    def train_on_batch_pairwise(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
        if self.multitask_mode == "lm_head":
            self._train_on_batch_pairwise_lm_head(inputs, labels)
            return
        self._train_on_batch_with_head(inputs, labels, head=self.pairwise_head)

    # ---------------------------------------------------------------------
    # LM-head pairwise implementation (no extra head)
    # ---------------------------------------------------------------------

    def _pairwise_sequence_loglikelihoods(self, prompts: List[str]) -> torch.Tensor:
        """Compute (B,3) log-likelihoods for choices \n[[1]]/\n[[2]]/\n[[3]].

        Returns:
            torch.Tensor on self.device with shape (B, 3).
        """

        if not prompts:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        # Build 3 variants per prompt and run in a single forward.
        texts: List[str] = []
        for p in prompts:
            base = str(p)
            for ct in self._pairwise_choice_texts:
                texts.append(base + ct)

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
            return_tensors="pt",
            add_special_tokens=True,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        attention_mask = batch.get("attention_mask")

        outputs = self.model(
            **batch,
            use_cache=False,
            return_dict=True,
        )
        logits: torch.Tensor = outputs.logits  # (B*3, L, V)

        # Score each sequence by summing token log-probs of the choice tokens.
        seq_scores: List[torch.Tensor] = []
        input_ids: torch.Tensor = batch["input_ids"]
        bsz = int(input_ids.shape[0])
        for i in range(bsz):
            # Effective length excludes padding.
            if attention_mask is not None:
                eff_len = int(attention_mask[i].sum().item())
            else:
                eff_len = int(input_ids.shape[1])

            ids_list = input_ids[i, :eff_len].tolist()

            # Determine which choice this row corresponds to.
            choice_idx = int(i % 3)
            sub = self._pairwise_choice_ids[choice_idx]
            start = self._find_subsequence_last(ids_list, sub)
            if start < 0:
                # Fallback: assume it is at the end (can happen when prompt is truncated).
                start = max(0, eff_len - len(sub))

            # Sum log p(token_t | prefix) for each token in the choice sequence.
            # token at position k is predicted by logits at k-1.
            s = torch.tensor(0.0, device=self.device)
            for j, tok_id in enumerate(sub):
                pos = int(start - 1 + j)
                if pos < 0 or pos >= eff_len - 1:
                    continue
                step_logits = logits[i, pos, :]
                # Avoid materializing a full (L,V) log-softmax tensor which can be >1GB.
                # log p(tok | context) = logit(tok) - logsumexp(logits)
                s = s + step_logits[int(tok_id)] - torch.logsumexp(step_logits, dim=-1)
            seq_scores.append(s)

        scores = torch.stack(seq_scores, dim=0).view(-1, 3)
        return scores

    def _predict_proba_pairwise_lm_head(self, inputs: Sequence[Any]) -> np.ndarray:
        inference_batch_size = 2
        out: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), inference_batch_size):
                batch_inputs = inputs[start : start + inference_batch_size]
                prompts = [str(x) for x in batch_inputs]
                ll = self._pairwise_sequence_loglikelihoods(prompts)  # (B,3)
                probs = torch.softmax(ll, dim=-1)
                out.append(probs.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.array([], dtype=np.float32)

    def _train_on_batch_pairwise_lm_head(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:
        mini_batch_size = 2
        n_samples = int(len(inputs))
        if n_samples <= 0:
            return

        self.model.train()
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, mini_batch_size):
            end = min(start + mini_batch_size, n_samples)
            batch_indices = indices[start:end]
            sub_inputs = [inputs[i] for i in batch_indices]
            sub_labels = torch.tensor([int(labels[i]) for i in batch_indices], dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()

            prompts = [str(x) for x in sub_inputs]

            ac = self._autocast_ctx()
            if ac is not None:
                with ac:
                    ll = self._pairwise_sequence_loglikelihoods(prompts)  # (B,3)
                    loss = torch.nn.functional.cross_entropy(ll, sub_labels)
            else:
                ll = self._pairwise_sequence_loglikelihoods(prompts)
                loss = torch.nn.functional.cross_entropy(ll, sub_labels)

            if getattr(self, "_scaler", None) is not None:
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            del sub_labels, ll, loss

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Backwards-compat shim (defaults to pointwise)
    # ---------------------------------------------------------------------

    def predict_proba(self, inputs: Sequence[Any]) -> np.ndarray:  # type: ignore[override]
        return self.predict_proba_pointwise(inputs)

    def train_on_batch(self, inputs: Sequence[Any], labels: Sequence[int]) -> None:  # type: ignore[override]
        self.train_on_batch_pointwise(inputs, labels)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _predict_proba_with_head(self, inputs: Sequence[Any], *, head: nn.Module) -> np.ndarray:
        # A 4k context on an 8B backbone can exceed a 40 GiB GPU with the
        # historical batch size of four. Keep the old throughput for shorter
        # contexts and use a conservative single-example batch for long ones.
        inference_batch_size = 1 if int(self.max_length) > 2048 else 4
        all_probs: list[np.ndarray] = []

        self.model.eval()
        head.eval()

        with torch.no_grad():
            for start in range(0, len(inputs), inference_batch_size):
                batch_inputs = inputs[start : start + inference_batch_size]
                features = self._extract_features(batch_inputs)
                logits = head(features)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)

    @staticmethod
    def _score_probs_from_threshold_probs(threshold_probs: torch.Tensor) -> torch.Tensor:
        if int(threshold_probs.shape[-1]) <= 0:
            raise ValueError("threshold_probs must have at least one threshold")
        threshold_probs = torch.cummin(threshold_probs.clamp(min=0.0, max=1.0), dim=-1).values
        prev = torch.cat(
            [
                torch.ones(
                    (threshold_probs.shape[0], 1),
                    dtype=threshold_probs.dtype,
                    device=threshold_probs.device,
                ),
                threshold_probs,
            ],
            dim=-1,
        )
        nxt = torch.cat(
            [
                threshold_probs,
                torch.zeros(
                    (threshold_probs.shape[0], 1),
                    dtype=threshold_probs.dtype,
                    device=threshold_probs.device,
                ),
            ],
            dim=-1,
        )
        probs = (prev - nxt).clamp(min=0.0)
        return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    def _predict_proba_with_coral_head(self, inputs: Sequence[Any], *, head: nn.Module) -> np.ndarray:
        inference_batch_size = 4
        all_probs: list[np.ndarray] = []

        self.model.eval()
        head.eval()

        with torch.no_grad():
            for start in range(0, len(inputs), inference_batch_size):
                batch_inputs = inputs[start : start + inference_batch_size]
                features = self._extract_features(batch_inputs)
                threshold_logits = head(features)
                threshold_probs = torch.sigmoid(threshold_logits)
                probs = self._score_probs_from_threshold_probs(threshold_probs)
                all_probs.append(probs.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)

    def _train_on_batch_with_head(self, inputs: Sequence[Any], labels: Sequence[int], *, head: nn.Module) -> None:
        self.model.train()
        head.train()

        mini_batch_size = 1 if int(self.max_length) > 2048 else 8
        n_samples = int(len(inputs))
        if n_samples <= 0:
            return

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, mini_batch_size):
            end = min(start + mini_batch_size, n_samples)
            batch_indices = indices[start:end]

            sub_inputs = [inputs[i] for i in batch_indices]
            sub_labels = [int(labels[i]) for i in batch_indices]

            batch = self._encode(sub_inputs)
            attention_mask = batch.get("attention_mask")
            labels_tensor = torch.tensor(sub_labels, dtype=torch.long, device=self.device)
            self.optimizer.zero_grad()

            ac = self._autocast_ctx()
            if ac is not None:
                with ac:
                    hidden_states = self._forward_last_hidden_state(batch)
                    features = self._pool_last_token_features(hidden_states, attention_mask)
                    logits = head(features)
                    loss = self.loss_fn(logits, labels_tensor)
            else:
                hidden_states = self._forward_last_hidden_state(batch)
                features = self._pool_last_token_features(hidden_states, attention_mask)
                logits = head(features)
                loss = self.loss_fn(logits, labels_tensor)

            if getattr(self, "_scaler", None) is not None:
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            del batch, labels_tensor, hidden_states, features, logits, loss

        # Try to release some temporary memory.
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _train_on_batch_with_coral_head(self, inputs: Sequence[Any], labels: Sequence[int], *, head: nn.Module) -> None:
        import torch.nn.functional as F

        self.model.train()
        head.train()

        mini_batch_size = 8
        n_samples = int(len(inputs))
        if n_samples <= 0:
            return

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, mini_batch_size):
            end = min(start + mini_batch_size, n_samples)
            batch_indices = indices[start:end]

            sub_inputs = [inputs[i] for i in batch_indices]
            sub_labels = [int(labels[i]) for i in batch_indices]

            batch = self._encode(sub_inputs)
            attention_mask = batch.get("attention_mask")
            labels_tensor = torch.tensor(sub_labels, dtype=torch.long, device=self.device)
            thresholds = torch.arange(
                0,
                int(self.pointwise_num_labels) - 1,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            targets = (thresholds < labels_tensor.to(dtype=torch.float32).unsqueeze(1)).to(dtype=torch.float32)
            self.optimizer.zero_grad()

            ac = self._autocast_ctx()
            if ac is not None:
                with ac:
                    hidden_states = self._forward_last_hidden_state(batch)
                    features = self._pool_last_token_features(hidden_states, attention_mask)
                    threshold_logits = head(features)
            else:
                hidden_states = self._forward_last_hidden_state(batch)
                features = self._pool_last_token_features(hidden_states, attention_mask)
                threshold_logits = head(features)

            loss = F.binary_cross_entropy_with_logits(
                threshold_logits.float(),
                targets,
                reduction="none",
            ).mean(dim=1)
            if self.pointwise_class_weights is not None:
                weights = self.pointwise_class_weights.to(self.device)[labels_tensor].float()
                loss = loss * weights
            loss = loss.mean()

            if getattr(self, "_scaler", None) is not None:
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            del batch, labels_tensor, hidden_states, features, threshold_logits, loss

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _train_on_batch_with_head_ordinal(self, inputs: Sequence[Any], labels: Sequence[int], *, head: nn.Module) -> None:
        import torch.nn.functional as F

        self.model.train()
        head.train()

        mini_batch_size = 8
        n_samples = int(len(inputs))
        if n_samples <= 0:
            return

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, mini_batch_size):
            end = min(start + mini_batch_size, n_samples)
            batch_indices = indices[start:end]

            sub_inputs = [inputs[i] for i in batch_indices]
            sub_labels = [int(labels[i]) for i in batch_indices]

            batch = self._encode(sub_inputs)
            attention_mask = batch.get("attention_mask")
            labels_tensor = torch.tensor(sub_labels, dtype=torch.long, device=self.device)
            thresholds = torch.arange(
                0,
                int(self.pointwise_num_labels) - 1,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            targets = (thresholds < labels_tensor.to(dtype=torch.float32).unsqueeze(1)).to(dtype=torch.float32)
            self.optimizer.zero_grad()

            ac = self._autocast_ctx()
            if ac is not None:
                with ac:
                    hidden_states = self._forward_last_hidden_state(batch)
                    features = self._pool_last_token_features(hidden_states, attention_mask)
                    logits = head(features)
                    probs = torch.softmax(logits, dim=-1)
                    threshold_probs = torch.flip(torch.cumsum(torch.flip(probs[:, 1:], dims=(1,)), dim=1), dims=(1,))
            else:
                hidden_states = self._forward_last_hidden_state(batch)
                features = self._pool_last_token_features(hidden_states, attention_mask)
                logits = head(features)
                probs = torch.softmax(logits, dim=-1)
                threshold_probs = torch.flip(torch.cumsum(torch.flip(probs[:, 1:], dims=(1,)), dim=1), dims=(1,))
            loss = F.binary_cross_entropy(
                threshold_probs.float().clamp(min=1e-6, max=1.0 - 1e-6),
                targets,
                reduction="none",
            ).mean(dim=1)
            if self.pointwise_class_weights is not None:
                weights = self.pointwise_class_weights.to(self.device)[labels_tensor].float()
                loss = loss * weights
            loss = loss.mean()

            if getattr(self, "_scaler", None) is not None:
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            del batch, labels_tensor, hidden_states, features, logits, loss

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

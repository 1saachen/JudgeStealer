# Ablation Experiment Checklist

## Frozen default method

The default candidate-selection profile is now:

```text
train_selection_mode = candidate_triple_selector
candidate_selector_kind = bias_trap_pointwise
diversity_weight = 1.0
uncertainty_weight = 0.25
bias_weight = 1.0
pointwise_length_bias_weight = 0.5
pairwise_position_bias_weight = 0.5
pairwise_position_bias_scale = 0.02
signal_normalization = none
uncertainty_view = pointwise
pointwise_global_smooth_alpha = 0.1
pointwise_global_smooth_mode = local_gaussian
```

Density remains a bottom-10% hard filter; coverage is diagnostic only and is not part of
the final acquisition score.

Random, no-smoothing, and other control runs must explicitly override
`--pointwise-global-smooth-alpha 0`; changing the method default must not silently add
smoothing to a control condition.

## Recommended execution order

Use Qwen3-1.7B full fine-tuning as the ablation anchor, with identical seed, split,
budget, stage schedule, prompt, and evaluation data. Run Alpaca and Dolly separately.

### 3.1 Impact of selector: 8 settings per dataset

- Random selection
- Without uncertainty (`uncertainty_weight=0`)
- Without diversity (`diversity_weight=0`)
- Without bias (`bias_weight=0`)
- Uncertainty only (`diversity_weight=bias_weight=0`)
- Diversity only (`uncertainty_weight=bias_weight=0`)
- Bias only (`diversity_weight=uncertainty_weight=0`)
- Hybrid selector (default profile)

### 3.2 Impact of smoothing: 6 settings per dataset

- No smoothing (`alpha=0`)
- Fixed `alpha=0.01`
- Fixed `alpha=0.05`
- Fixed `alpha=0.10` (default method)
- Fixed `alpha=0.20`
- Adaptive smoothing (pending implementation and a separate run)

### 3.3 Reviewing strategy: 2 settings per dataset

- Without reviewing
- Joint pointwise + pairwise reviewing

Before launching this block, freeze exactly which generated supervision is removed in
the "without reviewing" condition. The current script has no dedicated reviewing flag.

### 3.4 Query budget: 5 settings per dataset

- `B=0.5%`
- `B=1%`
- `B=2%`
- `B=5%`
- `B=10%`

The denominator for `B` must be fixed before launching. Convert each percentage to the
existing `budget_units` convention and keep initialization/batch proportions fixed.

## Run count

- Selector block: 8 x 2 datasets = 16
- Smoothing block: 6 x 2 datasets = 12
- Reviewing block: 2 x 2 datasets = 4
- Budget block: 5 x 2 datasets = 10
- Total anchor ablations: **42 runs**, plus adaptive smoothing once implemented.

After the anchor results are stable, repeat only the default profile for the remaining
surrogate sizes (Qwen3-0.6B, 4B, and 8B) and both datasets.

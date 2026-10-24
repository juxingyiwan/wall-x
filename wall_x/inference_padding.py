"""Inference-only virtual-channel policy; no model, CUDA or hardware imports.

``original`` preserves the upstream normalization result, including the
upstream short-statistics virtual-tail behavior. It never hardcodes -1.
``zero`` applies the project's training-aligned correction to named virtual
channels only. Masks, missing real joints and the initial noise stay intact.
"""


def validate_padding(value):
    if not isinstance(value, str) or value not in ("zero", "original"):
        raise ValueError("padding must be 'zero' or 'original'; use padding: zero")
    return value


def apply_inference_padding(values, channel_config, mode="zero"):
    mode = validate_padding(mode)
    if sum(int(size) for size in channel_config.values()) != values.shape[-1]:
        raise ValueError("channel_config width must match the input width")
    if mode == "original":
        return values
    # Lazy import keeps configuration inspection independent of torch/CUDA.
    from wall_x.model.core.action.normalizer import zero_virtual_padding

    return zero_virtual_padding(values, channel_config)

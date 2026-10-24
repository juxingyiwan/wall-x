import json
import logging

import torch
import torch.nn as nn

from wall_x.inference_padding import apply_inference_padding, validate_padding
from wall_x.utils.constant import _ACTION_KEY_FULL_MAPPING as _MODEL_KEY_TO_RAW_KEY

logger = logging.getLogger(__name__)


def resolve_normalizer_dataset_names(
    dataset_names,
    normalizer,
    batch_size,
    *,
    source="dataset_names",
    allow_skip_names=None,
):
    """Validate dataset names before indexing a normalizer ParameterDict.

    The normalizer key is data-dependent. Callers should pass the
    dataset names carried by the batch/model inputs, matching the qact/VLA
    path. This helper intentionally only validates presence, batch-size
    alignment, and key membership; it does not remap unknown names or fall
    back to a default key, since silent fallback can apply the wrong action
    scale without failing loudly. Pass ``allow_skip_names`` explicitly only
    for names that should bypass normalizer lookup, such as multimodal-only
    rows that do not carry action values.
    """
    if dataset_names is None:
        raise KeyError(f"Missing {source}; cannot choose normalizer key.")

    if isinstance(dataset_names, str):
        names = [dataset_names]
    elif isinstance(dataset_names, (list, tuple)):
        names = [str(name) for name in dataset_names]
    else:
        raise TypeError(
            f"{source} must be a string or sequence of strings, got "
            f"{type(dataset_names).__name__}"
        )

    if len(names) != batch_size:
        raise ValueError(
            f"{source} count ({len(names)}) does not match batch size "
            f"({batch_size}). names={names}"
        )

    available = list(getattr(normalizer, "delta", {}).keys())
    if not available:
        raise KeyError("Normalizer has no registered dataset keys.")

    allowed_skip = set(allow_skip_names or ())
    missing = sorted(
        {name for name in names if name not in available and name not in allowed_skip}
    )
    if missing:
        raise KeyError(
            f"{source} contains keys not present in normalizer: {missing}. "
            f"available={available}"
        )

    return names


def _normalizer_dataset_name_list(dataset_names):
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        return [dataset_names]
    return list(dataset_names)


def _normalizer_width(normalizer, dataset_names):
    names = _normalizer_dataset_name_list(dataset_names)
    if not names:
        return None
    if any(name not in normalizer.delta for name in names):
        return None
    widths = [int(normalizer.delta[name].shape[0]) for name in names]
    if not widths or len(set(widths)) != 1:
        return None
    return widths[0]


def _has_uniform_virtual_tail_mask(dof_mask, width):
    if dof_mask is None:
        return False
    mask = dof_mask.reshape(-1, dof_mask.shape[-1]).bool()
    if mask.shape[-1] <= width:
        return False
    prefix_active = mask[:, :width].all(dim=1)
    tail_inactive = (~mask[:, width:]).all(dim=1)
    return bool((prefix_active & tail_inactive).all().item())


def normalize_data_with_virtual_tail(normalizer, values, dataset_names, dof_mask):
    """Normalize real prefix dims when model tensors include a virtual tail."""
    width = _normalizer_width(normalizer, dataset_names)
    if width is None or values.shape[-1] == width:
        return normalizer.normalize_data(values, dataset_names)
    if values.shape[-1] < width or not _has_uniform_virtual_tail_mask(dof_mask, width):
        return normalizer.normalize_data(values, dataset_names)

    out = values.clone()
    out[..., :width] = normalizer.normalize_data(values[..., :width], dataset_names)
    out[..., width:] = 0
    return out


def unnormalize_data_with_virtual_tail(normalizer, values, dataset_names, dof_mask):
    """Unnormalize real prefix dims when model tensors include a virtual tail."""
    width = _normalizer_width(normalizer, dataset_names)
    if width is None or values.shape[-1] == width:
        return normalizer.unnormalize_data(values, dataset_names, None)
    if values.shape[-1] < width or not _has_uniform_virtual_tail_mask(dof_mask, width):
        return normalizer.unnormalize_data(values, dataset_names, dof_mask)

    out = values.clone()
    out[..., :width] = normalizer.unnormalize_data(
        values[..., :width], dataset_names, None
    )
    out[..., width:] = 0
    return out


# Padding alignment: training normalizes real values before appending virtual
# zeros. Enforce normalized zero even when checkpoint stats include padded slots.
def zero_virtual_padding(values, channel_config):
    """Zero only explicit ``action_padding`` slots in a tensor or ndarray.

    LeRobot appends these slots after normalization. A zero mask can also
    describe an unavailable real joint, so masks alone must not identify
    virtual padding. Preserve real channels and never mutate the input.
    """
    if sum(int(size) for size in channel_config.values()) != values.shape[-1]:
        raise ValueError("channel_config width must match the input width")
    if "action_padding" not in channel_config:
        return values
    result = values.clone() if isinstance(values, torch.Tensor) else values.copy()
    offset = 0
    for channel, size in channel_config.items():
        end = offset + int(size)
        if channel == "action_padding":
            result[..., offset:end] = 0
        offset = end
    return result


# The mask is not an attention barrier: the full action vector is encoded
# jointly, so a wrong virtual endpoint can affect real-joint outputs. Zero is
# the clean endpoint, not the whole denoising trajectory: keep sampled noise
# and the flow path from that noise to zero. Do not rewrite checkpoint stats.
# The same policy is used by state preprocessing and native/RTC flow sampling.
def build_flow_padding_action(
    normalizer, dataset_names, dof_mask, dof_config, *, mode="zero"
):
    """Build normalized endpoints for inactive flow-action dimensions.

    LeRobot's collator appends ``action_padding`` *after* normalization, so
    its endpoint is normalized zero. Normalizing a raw zero with legacy
    padded stats (min=0, delta=1) would instead produce -1. These dimensions
    still enter the action encoder even though their loss is masked out.

    Use the explicit channel layout to distinguish virtual padding from
    missing real joints; missing real joints retain their normalizer's
    existing raw-zero endpoint. ``mode="original"`` returns the upstream
    normalization result before the virtual-zero correction. No checkpoint
    statistics are modified.
    """
    mode = validate_padding(mode)
    action_dim = dof_mask.shape[-1]
    if sum(int(size) for size in dof_config.values()) != action_dim:
        raise ValueError("dof_config width must match the flow-action mask")

    batch_size = dof_mask.shape[0]
    names = resolve_normalizer_dataset_names(
        dataset_names, normalizer, batch_size, source="flow dataset_names"
    )
    padding = torch.zeros(
        (batch_size, 1, action_dim), device=dof_mask.device, dtype=torch.float32
    )
    padding = normalize_data_with_virtual_tail(normalizer, padding, names, dof_mask)

    offset = 0
    for channel, size in dof_config.items():
        end = offset + int(size)
        if channel == "action_padding":
            if dof_mask[..., offset:end].any():
                raise ValueError("action_padding must be inactive in dof_mask")
        offset = end
    return apply_inference_padding(padding, dof_config, mode)


def print_rank_last(message):
    """If distributed is initialized, log only on last rank."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            logger.info(message)
    else:
        logger.info(message)


class Normalizer(nn.Module):
    @classmethod
    def from_ckpt(cls, ckpt_path):
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.min = nn.ParameterDict()
        instance.delta = nn.ParameterDict()
        instance.min_key = "min"
        instance.delta_key = "delta"

        ckpt = torch.load(ckpt_path, map_location="cpu")

        for key, value in ckpt.items():
            # parse key: "min.robot_name" -> prefix="min", name="robot_name"
            try:
                prefix, name = key.split(".", 1)
                if hasattr(instance, prefix):
                    getattr(instance, prefix)[name] = nn.Parameter(
                        value, requires_grad=False
                    )
            except ValueError:
                continue

        return instance

    @classmethod
    def from_lerobot_norm_stats(
        cls,
        action_stats,
        dataset_name,
    ):

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Set containers
        instance.min = nn.ParameterDict()
        instance.delta = nn.ParameterDict()

        # Fill dataset entry
        instance.min[dataset_name] = nn.Parameter(action_stats.min, requires_grad=False)
        instance.delta[dataset_name] = nn.Parameter(
            action_stats.delta, requires_grad=False
        )

        # Record keys
        instance.min_key = "min"
        instance.delta_key = "delta"

        return instance

    def __init__(
        self,
        action_statistic_dof,
        dof_config,
        min_key="min",
        delta_key="delta",
        name="normalizer",
    ):
        super(Normalizer, self).__init__()

        self.min_key = min_key
        self.delta_key = delta_key

        action_statistic = {}
        normalizer_missing_information = []
        for robot_name in action_statistic_dof.keys():
            action_statistic[robot_name] = {}
            all_dof_min = []
            all_dof_delta = []
            for k in dof_config:
                if (
                    k not in action_statistic_dof[robot_name]
                    and k.replace("master_", "follow_", 1)
                    in action_statistic_dof[robot_name]
                ):
                    k = k.replace("master_", "follow_", 1)
                if k in action_statistic_dof[robot_name]:
                    if (
                        min_key in action_statistic_dof[robot_name][k]
                        and delta_key in action_statistic_dof[robot_name][k]
                    ):
                        all_dof_min.extend(action_statistic_dof[robot_name][k][min_key])
                        all_dof_delta.extend(
                            action_statistic_dof[robot_name][k][delta_key]
                        )
                    else:
                        normalizer_missing_information.append(
                            f"Normalizer (Warning): min_key {min_key} or delta_key {delta_key} not in action_statistic_dof[{robot_name}][{k}], use default min 0.0 and delta 1.0"
                        )
                        all_dof_min.extend([0.0] * dof_config[k])
                        all_dof_delta.extend([1.0] * dof_config[k])
                else:
                    # k is a model key; action_statistic_dof may store raw keys - try fallback lookup
                    raw_key = _MODEL_KEY_TO_RAW_KEY.get(k)
                    if (
                        raw_key is not None
                        and raw_key in action_statistic_dof[robot_name]
                    ):
                        stat = action_statistic_dof[robot_name][raw_key]
                        if min_key in stat and delta_key in stat:
                            all_dof_min.extend(stat[min_key])
                            all_dof_delta.extend(stat[delta_key])
                        else:
                            normalizer_missing_information.append(
                                f"Normalizer (Warning): Action {k} not in action_statistic_dof for {robot_name}, use default min 0.0 and delta 1.0"
                            )
                            all_dof_min.extend([0.0] * dof_config[k])
                            all_dof_delta.extend([1.0] * dof_config[k])
                    else:
                        normalizer_missing_information.append(
                            f"Normalizer (Warning): Action {k} not in action_statistic_dof for {robot_name}, use default min 0.0 and delta 1.0"
                        )
                        all_dof_min.extend([0.0] * dof_config[k])
                        all_dof_delta.extend([1.0] * dof_config[k])
            all_dof_min = torch.tensor(all_dof_min)
            all_dof_delta = torch.tensor(all_dof_delta)
            action_statistic[robot_name][min_key] = all_dof_min
            action_statistic[robot_name][delta_key] = all_dof_delta

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1
        ):
            # export normalizer_missing_information to file
            with open(f"normalizer_missing_information_{name}.txt", "w") as f:
                for info in normalizer_missing_information:
                    f.write(info + "\n")
            logger.info(
                "Normalizer missing information saved to normalizer_missing_information_%s.txt",
                name,
            )

        self.min = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k][min_key], requires_grad=False)
                for k in action_statistic.keys()
            }
        )
        self.delta = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k][delta_key], requires_grad=False)
                for k in action_statistic.keys()
            }
        )

    def normalize_data(self, xs, dataset_names):
        new_xs = []
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        for x, dataset_name in zip(xs, dataset_names):
            if getattr(self, "affine_fp32", False):
                from wall_x.model.core.action.affine import scale

                new_xs.append(
                    scale(
                        x,
                        self.min[dataset_name],
                        self.delta[dataset_name],
                        clip=self.clip,
                    )
                )
                continue
            delta = self.delta[dataset_name]
            delta = torch.where(delta == 0, torch.ones_like(delta), delta)
            x = (x - self.min[dataset_name]) / delta
            x = x * 2 - 1
            x = torch.clamp(x, -1, 1)
            new_xs.append(x)
        new_xs = torch.stack(new_xs)
        return new_xs

    def unnormalize_data(self, xs, dataset_names, dof_mask=None):
        new_xs = []
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        dof_mask = dof_mask if dof_mask is not None else [None] * len(xs)
        for x, dataset_name, mask in zip(xs, dataset_names, dof_mask):
            if getattr(self, "affine_fp32", False):
                from wall_x.model.core.action.affine import unscale

                lower, span = self.min[dataset_name], self.delta[dataset_name]
                if mask is not None:
                    lower, span = lower[mask[0].bool()], span[mask[0].bool()]
                new_xs.append(unscale(x, lower, span))
                continue
            x = (x + 1) / 2
            if mask is not None:
                mask = mask[0].bool()
                action_space_delta = self.delta[dataset_name][mask]
                action_space_min = self.min[dataset_name][mask]
            else:
                action_space_delta = self.delta[dataset_name]
                action_space_min = self.min[dataset_name]
            x = x * action_space_delta + action_space_min
            new_xs.append(x)
        new_xs = torch.stack(new_xs)
        return new_xs


def _pad_lerobot_stats(stats, target_dim, label):
    """Tail-pad flat LeRobot stats to the model's configured action/state dim."""
    target_dim = int(target_dim)
    current_dim = int(stats.min.numel())
    if current_dim == target_dim:
        return stats
    if current_dim > target_dim:
        raise ValueError(
            f"LeRobot {label} norm stats dim ({current_dim}) exceeds configured "
            f"{label} dim ({target_dim})"
        )

    pad_dim = target_dim - current_dim
    min_stat = torch.cat(
        [stats.min.flatten(), torch.zeros(pad_dim, dtype=stats.min.dtype)]
    )
    delta = torch.cat(
        [stats.delta.flatten(), torch.ones(pad_dim, dtype=stats.delta.dtype)]
    )
    max_stat = torch.cat(
        [stats.max.flatten(), torch.ones(pad_dim, dtype=stats.max.dtype)]
    )
    logger.info(
        "Padded LeRobot %s normalizer stats from %d to %d dims",
        label,
        current_dim,
        target_dim,
    )
    return type(stats)(min=min_stat, max=max_stat, delta=delta)


def pad_normalizer_to_dim(normalizer, target_dim, label):
    """Tail-pad every dataset entry to ``target_dim`` (virtual ``action_padding`` tail)."""
    target_dim = int(target_dim)
    for name in list(normalizer.min.keys()):
        current_dim = int(normalizer.min[name].numel())
        if current_dim == target_dim:
            continue
        if current_dim > target_dim:
            raise ValueError(
                f"Normalizer {label}[{name!r}] dim ({current_dim}) exceeds configured "
                f"{label} dim ({target_dim})"
            )

        pad_dim = target_dim - current_dim
        dtype = normalizer.min[name].dtype
        normalizer.min[name] = nn.Parameter(
            torch.cat(
                [
                    normalizer.min[name].flatten(),
                    torch.zeros(pad_dim, dtype=dtype),
                ]
            ),
            requires_grad=False,
        )
        normalizer.delta[name] = nn.Parameter(
            torch.cat(
                [
                    normalizer.delta[name].flatten(),
                    torch.ones(pad_dim, dtype=dtype),
                ]
            ),
            requires_grad=False,
        )
        logger.info(
            "Padded %s normalizer[%r] from %d to %d dims",
            label,
            name,
            current_dim,
            target_dim,
        )


def create_normalizers_from_lerobot_norm_stats(
    norm_stats, dataset_name, action_dim, propri_dim
):
    """Create model normalizers from LeRobot flat action/state norm stats."""
    action_stats = _pad_lerobot_stats(norm_stats["action"], action_dim, "action")
    propri_stats = _pad_lerobot_stats(norm_stats["state"], propri_dim, "state")
    return (
        Normalizer.from_lerobot_norm_stats(action_stats, dataset_name),
        Normalizer.from_lerobot_norm_stats(propri_stats, dataset_name),
    )


def create_normalizers(config, action_statistic_dof=None):
    """Create action and proprioception normalizers from explicit stats.

    Normalization statistics must come from the config, checkpoint, or dataset.
    Public Wall-X builds intentionally do not bundle private default stats.
    """

    if action_statistic_dof is None:
        custom_path = config.get("customized_action_statistic_dof")
        if custom_path:
            with open(custom_path, "r") as f:
                action_statistic_dof = json.load(f)
        else:
            raise ValueError(
                "create_normalizers requires action statistics. Provide "
                "`customized_action_statistic_dof` in the config or pass "
                "`action_statistic_dof` from the checkpoint/dataset."
            )

    min_key = config.get("min_key", "min")
    delta_key = config.get("delta_key", "delta")

    # Required keys for constructing action/proprio normalizers.
    missing_required = []
    if config.get("dof_config") is None:
        missing_required.append("dof_config")
    if config.get("agent_pos_config") is None:
        missing_required.append("agent_pos_config")
    if missing_required:
        raise KeyError(
            "create_normalizers requires non-None config keys: "
            + ", ".join(missing_required)
        )

    normalizer_action = Normalizer(
        action_statistic_dof,
        config["dof_config"],
        min_key=min_key,
        delta_key=delta_key,
    )
    normalizer_propri = Normalizer(
        action_statistic_dof,
        config["agent_pos_config"],
        min_key=min_key,
        delta_key=delta_key,
    )
    return normalizer_action, normalizer_propri, action_statistic_dof

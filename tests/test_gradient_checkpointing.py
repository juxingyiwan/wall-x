"""Checkpoint recomputation must support the vision block's keyword inputs."""

import torch
from transformers import PretrainedConfig, PreTrainedModel

from wall_x.trainer.fsdp_trainer.distribution_strategy import FSDPStrategy


class KeywordVisionModel(PreTrainedModel):
    config_class = PretrainedConfig
    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__(PretrainedConfig())
        self.gradient_checkpointing = False
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def block(self, hidden_states, *, position_embeddings):
        return (hidden_states + position_embeddings) * self.weight

    def forward(self, hidden_states, position_embeddings):
        return self._gradient_checkpointing_func(
            self.block, hidden_states, position_embeddings=position_embeddings
        )


def test_checkpointed_vision_keyword_inputs_backpropagate():
    model = KeywordVisionModel()
    strategy = FSDPStrategy({"use_gradient_checkpointing": True})
    strategy._maybe_enable_grad_ckpt(model)
    # Frozen image inputs need not require grad; trainable block weights do.
    inputs = torch.tensor([1.0, 2.0])
    positions = torch.tensor([3.0, 4.0])
    actual = model(inputs, positions)
    torch.testing.assert_close(actual, torch.tensor([8.0, 12.0]))
    actual.sum().backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor(10.0))

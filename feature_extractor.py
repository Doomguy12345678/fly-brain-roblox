"""Compact convolutional feature extraction for image-dictionary observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ScreenFeatureExtractor(BaseFeaturesExtractor):
    """Apply a small CNN to each visual observation channel group."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 384,
    ) -> None:
        keys = ("frame", "difference", "history")
        if features_dim < len(keys) or features_dim % len(keys) != 0:
            raise ValueError("features_dim must be positive and divisible by three")
        feature_per_input = features_dim // len(keys)
        super().__init__(observation_space, features_dim=feature_per_input * len(keys))

        self.extractors = nn.ModuleDict()
        for key in keys:
            shape: Sequence[int] = observation_space[key].shape
            if len(shape) != 3:
                raise ValueError(f"Expected an HWC image for observation {key!r}")
            channels = shape[-1]
            self.extractors[key] = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(64 * 2 * 2, feature_per_input),
                nn.ReLU(),
            )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        extracted = [
            self.extractors[key](
                observations[key].permute(0, 3, 1, 2).float() / 255.0
            )
            for key in ("frame", "difference", "history")
        ]
        return torch.cat(extracted, dim=1)

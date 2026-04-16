from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WeatherGRUExtractor(BaseFeaturesExtractor):
    """
    Split the observation into:
    - non-forecast state features -> MLP
    - contiguous forecast sequence -> GRU

    This matches the Article 22 idea more closely than a plain MLP:
    predictive weather information is processed as a short sequence rather than
    as unrelated scalar features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        forecast_start: int,
        forecast_len: int,
        state_hidden: int = 128,
        gru_hidden: int = 32,
        combined_hidden: int = 128,
        features_dim: int = 128,
    ):
        obs_dim = int(observation_space.shape[0])
        if forecast_start < 0 or forecast_len <= 0 or forecast_start + forecast_len > obs_dim:
            raise ValueError(
                f"Invalid forecast slice: start={forecast_start}, len={forecast_len}, obs_dim={obs_dim}"
            )

        self.forecast_start = int(forecast_start)
        self.forecast_len = int(forecast_len)
        self.state_dim = obs_dim - self.forecast_len
        super().__init__(observation_space, features_dim)

        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden),
            nn.ReLU(),
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.combined_net = nn.Sequential(
            nn.Linear(state_hidden + gru_hidden, combined_hidden),
            nn.ReLU(),
            nn.Linear(combined_hidden, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        forecast = observations[:, self.forecast_start : self.forecast_start + self.forecast_len]
        state = torch.cat(
            [
                observations[:, : self.forecast_start],
                observations[:, self.forecast_start + self.forecast_len :],
            ],
            dim=1,
        )

        state_feat = self.state_net(state)
        forecast_seq = forecast.unsqueeze(-1)
        _, hidden = self.gru(forecast_seq)
        forecast_feat = hidden.squeeze(0)

        return self.combined_net(torch.cat([state_feat, forecast_feat], dim=1))

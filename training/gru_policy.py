

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class WeatherGRUExtractor(BaseFeaturesExtractor):
    

    def __init__(
        self,
        observation_space: spaces.Box,
        state_dim: int = 4,
        forecast_len: int = 24,
        state_hidden: int = 64,
        gru_hidden: int = 32,
        combined_hidden: int = 64,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.state_dim = state_dim
        self.forecast_len = forecast_len

        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
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
        
        state = observations[:, :self.state_dim]                    
        weather = observations[:, self.state_dim:]                  

        
        state_feat = self.state_net(state)                          

        
        weather_seq = weather.unsqueeze(-1)                         
        _, gru_hidden = self.gru(weather_seq)                       
        weather_feat = gru_hidden.squeeze(0)                        

        
        combined = torch.cat([state_feat, weather_feat], dim=1)     
        return self.combined_net(combined)
                              
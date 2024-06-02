import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Custom LSTM feature extractor
class CustomLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomLSTMExtractor, self).__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(input_size=observation_space.shape[0], hidden_size=64, batch_first=True)
        self.linear = nn.Linear(64, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        lstm_out, _ = self.lstm(observations.unsqueeze(0))  # Adding batch dimension
        lstm_out = lstm_out.squeeze(0)  # Removing batch dimension
        return self.linear(lstm_out)


# Custom LSTM policy
class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, lr_schedule, net_arch=None, **kwargs):
        super(CustomLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, **kwargs)
        self.features_extractor = CustomLSTMExtractor(observation_space, features_dim=64)
        self.mlp_extractor = self._build_mlp_extractor()
        self.action_net = nn.Linear(64, action_space.n)
        self.value_net = nn.Linear(64, 1)

    def _build_mlp_extractor(self):
        return nn.Sequential(
            nn.Linear(self.features_extractor.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, distribution.log_prob(actions)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(features)
        return values, log_prob, entropy

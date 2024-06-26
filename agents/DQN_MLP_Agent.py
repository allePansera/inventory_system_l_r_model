from stable_baselines3 import DQN
from agents.AgentAbs import Agent
from agents.Callback import RewardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import tensorflow as tf


class DqnMlp(Agent):
    """
    Param:

    policy:

        Description: This defines the policy model used by the agent. It can be a feed-forward neural network (MLP), a convolutional neural network (CNN), or a recurrent neural network (LSTM), depending on the environment and the type of observations.
        Default: 'MlpPolicy' (for environments with low-dimensional observations)

    learning_rate:

        Description: The learning rate controls how much the model's weights are adjusted with respect to the loss gradient. A higher learning rate means faster learning, but it might overshoot the optimal values. Conversely, a lower learning rate means slower but potentially more precise updates.
        Default: 0.0005

    buffer_size:

        Description: The size of the replay buffer, which stores the agent's experiences (state, action, reward, next state, done). A larger buffer size allows the agent to learn from a more diverse set of experiences.
        Default: 50000

    exploration_fraction:

        Description: The fraction of the total training steps during which the exploration rate (epsilon in the epsilon-greedy policy) is linearly decreased from its initial value to its final value. It determines how long the agent will explore before focusing on exploitation.
        Default: 0.1

    exploration_final_eps:

        Description: The final value of epsilon after the exploration phase. It defines the probability of taking a random action instead of the best-known action.
        Default: 0.02

    train_freq:

        Description: The frequency (in terms of steps) with which the model is updated. For example, if train_freq=1, the model is updated at every step. If train_freq=4, the model is updated every four steps.
        Default: 1

    batch_size:

        Description: The number of experiences sampled from the replay buffer for each update of the model. A larger batch size can lead to more stable updates but requires more memory.
        Default: 32

    target_network_update_freq:

        Description: The frequency (in terms of steps) with which the target network is updated to match the weights of the main network. This helps stabilize training by keeping the target values more consistent.
        Default: 500

    gamma:

        Description: The discount factor for future rewards. It determines the importance of future rewards compared to immediate rewards. A gamma close to 1 makes the agent focus on long-term rewards, while a gamma close to 0 makes it focus on short-term rewards.
        Default: 0.99

    learning_starts:

        Description: The number of steps before the agent starts learning. This allows the replay buffer to collect a sufficient amount of experiences before updates begin.
        Default: 1000

    target_network_update_interval:

        Description: Similar to target_network_update_freq, it specifies the interval at which the target network's weights are updated.
        Default: None (not used if target_network_update_freq is specified)

    prioritized_replay:

        Description: A boolean parameter that indicates whether prioritized experience replay is used. When set to True, it allows the agent to prioritize important experiences over less important ones, improving learning efficiency.
        Default: False

    prioritized_replay_alpha:

        Description: The exponent determining the degree of prioritization. A value of 0 means no prioritization, and a value closer to 1 means higher prioritization of important experiences.
        Default: 0.6

    policy_kwargs = dict(
        net_arch=[64, 64]  # Two hidden layers with 64 units each
    )
    """

    def __init__(self):
        self.model = None
        self.w_env = None
        self.reward_callback = None
        self.checkpoint_callback = None
        self.id = "dqn_mlp"

    def train(self, w_env: gym.Env, episode_duration=1000, plot_rewards=True, use_params=True):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param episode_duration: how many episode run to find optimal policy and value function
        :param plot_rewards: choose whether plot or not reward progression during training
        :param use_params: choose whether use or not default params
        :return:
        """
        params = {
            'learning_rate': 0.0005,        # α — default: 0.0005
            'buffer_size': 1000000,         # size of the replay buffer — default: 1000000
            'learning_starts': 100,         # how many steps of the model to collect transitions for before learning starts — default: 100
            "batch_size": 256,              # Size of the batch — default: 32
            'tau': 1.0,                     # target network update frequency — default: 1.0
            'gamma': 0.99,                  # discount factor — default: 0.99
            'train_freq': 4,                # how often the training step is done — default: 4
            'gradient_steps': 1,            # how many gradient steps to do after each rollout — default: 1
            'target_update_interval': 32,   # update the target network every `x` steps — default: 10000
            'exploration_fraction': 0.1,    # fraction of entire training period over which the exploration rate is reduced — default: 0.1
            'exploration_initial_eps': 1,   # ε (initial) — default: 1.0
            'exploration_final_eps': 0.15,  # ε (final) — default: 0.05
            "policy_kwargs": {              # Additional arguments to be passed to the policy on creation
                "net_arch": [               # Custom network architecture
                    128, 128
                ]
            },
        }
        self.w_env = w_env
        if use_params:
            self.model = DQN("MlpPolicy", self.w_env, verbose=0, tensorboard_log="./log/dqn_mlp_tensorboard", **params)
        else:
            self.model = DQN("MlpPolicy", self.w_env, verbose=0, tensorboard_log="./log/dqn_mlp_tensorboard")
        self.reward_callback = RewardCallback("DQN")
        self.checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path="models/checkpoints",
            name_prefix=self.id
        )
        self.model.learn(total_timesteps=episode_duration,
                         callback=[self.reward_callback, self.checkpoint_callback]
                         )

    def predict(self, observation):
        """
        Need to call train() before that
        :param observation: list of all states (actually 6)
        :return: action and new states
        """
        assert self.model is not None
        return self.model.predict(observation)

    def save_model(self, path: str = "models/dqn_mlp"):
        """
        Save trained model
        :param path: location to use to read model paramters
        :return:
        """
        assert self.model is not None
        self.model.save(path)

    def load_model(self, w_env: gym.Env, path: str = "models/dqn_mlp"):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param path: location to use to read model paramters
        :return:
        """
        self.w_env = w_env
        self.model = DQN('MlpPolicy', self.w_env, verbose=0)
        self.model.load(path)

    def __repr__(self):
        return "DQN - MLP Policy"

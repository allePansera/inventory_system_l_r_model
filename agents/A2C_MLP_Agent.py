from stable_baselines3 import A2C
from agents.AgentAbs import Agent
from agents.Callback import RewardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import tensorflow as tf


class A2cMlp(Agent):
    """
    Param:

    policy:
        'MlpPolicy': Uses a multi-layer perceptron (MLP) for the policy network.

    learning_rate:
        0.0007: A default learning rate which balances the speed and stability of training.

    n_steps:
        5: The agent collects data for 5 steps before performing an update.

    gamma:
        0.99: The discount factor, which prioritizes future rewards slightly less than immediate rewards.

    gae_lambda:
        1.0: Uses the Generalized Advantage Estimation with full weight on the advantage function.

    ent_coef:
        0.0: No extra exploration encouragement by default.

    vf_coef:
        0.25: A balanced weight for the value function loss in the total loss calculation.

    max_grad_norm:
        0.5: Clips the gradients to prevent excessive updates.

    rms_prop_eps:
        1e-5: A small value to ensure numerical stability in the RMSProp optimizer.

    use_rms_prop:
        True: Uses the RMSProp optimizer, which can be beneficial for certain environments.

    normalize_advantage:
        False: No normalization of advantages by default.

    tensorboard_log:
        None: No logging to TensorBoard by default.

    create_eval_env:
        False: No separate evaluation environment is created by default.

    verbose:
        0: No output unless there is an error or warning.

    device:
        'auto': Automatically uses a GPU if available.

    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Two hidden layers with 64 units each for both policy and value function
    )

    """

    def __init__(self):
        self.model = None
        self.w_env = None
        self.reward_callback = None
        self.checkpoint_callback = None
        self.id = "a2c_mlp"

    def train(self, w_env: gym.Env, episode_duration=1000, plot_rewards=True, use_params=True):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param episode_duration: how many episode run to find optimal policy and value function
        :param plot_rewards: choose whether plot or not reward progression during training
        :param use_params: choose whether use or not params
        :return:
        """
        self.w_env = w_env
        params = {
            'learning_rate': 0.05,         # α — default: 0.0007
            'n_steps': 10,                 # number of steps to run for each environment per update — default: 5
            'gamma': 0.99,                 # discount factor — default: 0.99
            'gae_lambda': 0.95,            # λ - Generalized Advantage Estimator — default: 1.0
            'ent_coef': 0.0,               # entropy coefficient — default: 0.0
            'vf_coef': 1.0,                # value function coefficient — default: 0.5
            'max_grad_norm': 2.0,          # max gradient norm — default: 0.5
            'normalize_advantage': False,  # normalize advantage — default: False
            "policy_kwargs": {             # Additional arguments to be passed to the policy on creation
                "net_arch": [              # Custom network architecture
                    dict(pi=[128, 128], vf=[128, 128])
                ]
            }
        }
        if use_params:
            self.model = A2C("MlpPolicy", self.w_env, verbose=0, tensorboard_log="./log/a2c_mlp_tensorboard", **params)
        else:
            self.model = A2C("MlpPolicy", self.w_env, verbose=0, tensorboard_log="./log/a2c_mlp_tensorboard")
        self.reward_callback = RewardCallback("A2C")
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

    def save_model(self, path: str = "models/a2c_mlp"):
        """
        Save trained model
        :param path: location to use to read model paramters
        :return:
        """
        assert self.model is not None
        self.model.save(path)

    def load_model(self, w_env: gym.Env, path: str = "models/a2c_mlp"):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param path: location to use to read model paramters
        :return:
        """
        self.w_env = w_env
        self.model = A2C('MlpPolicy', self.w_env, verbose=0)
        self.model.load(path)

    def __repr__(self):
        return "A2C - MLP Policy"

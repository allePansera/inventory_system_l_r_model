from stable_baselines3 import PPO
from agents.AgentAbs import Agent
from agents.Policy import CustomLSTMExtractor
from agents.Callback import RewardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym


class PpoMlp(Agent):
    """
    Pseudo Code:

    Initialize policy network πθ with parameters θ
    Initialize value function Vθv with parameters θv
    Initialize replay buffer

    for each iteration do:
        for each environment step do:
            Collect observation st
            Sample action at from policy πθ(at | st)
            Execute action at and observe reward rt and next state st+1
            Store (st, at, rt, st+1) in replay buffer
        end for

        for each update step do:
            Sample a random minibatch of transitions (st, at, rt, st+1) from the replay buffer

            Compute advantages using Generalized Advantage Estimation (GAE)
            δt = rt + γ * Vθv(st+1) - Vθv(st)
            At = δt + γ * λ * δt+1 + (γ * λ)^2 * δt+2 + ...

            Compute the ratio of new and old policy probabilities
            rt(θ) = πθ(at | st) / πθ_old(at | st)

            Compute the surrogate loss for the policy
            L_clip(θ) = E[min(rt(θ) * At, clip(rt(θ), 1 - ε, 1 + ε) * At)]

            Compute the value function loss
            L_vf(θv) = (Vθv(st) - Rt)^2

            Compute the entropy bonus to encourage exploration
            S[πθ](st) = -Σ πθ(at | st) * log(πθ(at | st))

            Compute the total loss
            L(θ, θv) = L_clip(θ) - c1 * L_vf(θv) + c2 * S[πθ](st)

            Perform gradient descent on L(θ, θv) to update θ and θv

        end for
    end for

    Param:

    policy:

        Description: This defines the policy model used by the agent. It can be a feed-forward neural network (MLP), a convolutional neural network (CNN), or a recurrent neural network (LSTM), depending on the environment and the type of observations.
        Default: 'MlpPolicy' (for environments with low-dimensional observations)

    learning_rate:

        Description: The learning rate controls how much the model's weights are adjusted with respect to the loss gradient. A higher learning rate means faster learning, but it might overshoot the optimal values. Conversely, a lower learning rate means slower but potentially more precise updates.
        Default: 3e-4

    n_steps:

        Description: The number of steps to run for each environment per update. The total batch size is n_steps * n_envs where n_envs is the number of parallel environments.
        Default: 2048

    batch_size:

        Description: The number of experiences sampled from the replay buffer for each update of the model. A larger batch size can lead to more stable updates but requires more memory.
        Default: 64

    n_epochs:

        Description: The number of epochs to update the policy. For each update, the policy is trained for n_epochs on the batch sampled from the replay buffer.
        Default: 10

    gamma:

        Description: The discount factor for future rewards. It determines the importance of future rewards compared to immediate rewards. A gamma close to 1 makes the agent focus on long-term rewards, while a gamma close to 0 makes it focus on short-term rewards.
        Default: 0.99

    gae_lambda:

        Description: The lambda parameter for Generalized Advantage Estimation. This controls the trade-off between bias and variance in the advantage estimation.
        Default: 0.95

    clip_range:

        Description: The clipping parameter for the PPO objective. It clips the probability ratio between the new and old policies to ensure the updates do not deviate too much from the old policy.
        Default: 0.2

    clip_range_vf:

        Description: The clipping parameter for the value function loss. If set to None, no clipping is applied to the value function.
        Default: None

    ent_coef:

        Description: The coefficient for the entropy term in the loss function. Entropy regularization is used to encourage exploration by penalizing high certainty (low entropy).
        Default: 0.0

    vf_coef:

        Description: The coefficient for the value function term in the loss function. This term is used to scale the contribution of the value function loss.
        Default: 0.5

    max_grad_norm:

        Description: The maximum value for the gradient clipping. This helps in stabilizing training by preventing the gradients from becoming too large.
        Default: 0.5

    use_sde:

        Description: Whether to use State Dependent Exploration (SDE) instead of the default Gaussian exploration strategy.
        Default: False

    sde_sample_freq:

        Description: The frequency of sampling the noise matrix for SDE. If set to -1, the noise is sampled at each step.
        Default: -1

    target_kl:

        Description: The target value for the KL divergence between the old and new policies. If the KL divergence exceeds this value, training is stopped early.
        Default: None
    """
    def __init__(self):
        self.model = None
        self.w_env = None
        self.reward_callback = None
        self.checkpoint_callback = None
        self.id = "ppo_mlp"

    def train(self, w_env: gym.Env, episode_duration=1000, plot_rewards=True):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param episode_duration: how many episode run to find optimal policy and value function
        :param plot_rewards: choose whether plot or not reward progression during training
        :return:
        """
        self.w_env = w_env
        self.model = PPO("MlpPolicy", self.w_env, verbose=0)
        self.reward_callback = RewardCallback("PPO")
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

    def save_model(self, path: str = "models/ppo_mlp"):
        """
        Save trained model
        :param path: location to use to read model paramters
        :return:
        """
        assert self.model is not None
        self.model.save(path)

    def load_model(self,  w_env: gym.Env, path: str = "models/ppo_mlp"):
        """
        Train the agent.
        :param w_env: gym Environment instance
        :param path: location to use to read model paramters
        :return:
        """
        self.w_env = w_env
        self.model = PPO('MlpPolicy', self.w_env, verbose=1)
        self.model.load(path)

    def __repr__(self):
        return "PPO - MLP Policy"



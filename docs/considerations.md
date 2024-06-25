# Alessandro Pansera/Mattia Bertacchini - Inventory System
## System's input description
- Project has been realized considering day as UoM. Suppose 1 Month = 30 Days for conversion.
- Items are apple's products... not my personal choice...

## 2^k Factorial Design
### Considerations for the s-d system

On average, obtained best results for the system are:

- [Item 0: s=20, S=42 (d=22)](2k_factorial_design/item_0_analysis.png)
- [Item 1: s=6, S=27 (d=21)](2k_factorial_design/item_0_analysis.png)

## Result

### 1.

Test using reward and observations normalizations.

```log
Training agent: A2C - MLP Policy
Loaded model from models/checkpoints/a2c_mlp_truncated.zip
Training started...
Training Progress:  22%|██▏       | 218446/1000000 [1:30:03<11:13:42, 19.33it/s]
```
### 2. Using same hyperparameters

Test without reward and observations normalizations.
Custom parameters have been used.

```json
{
  "a2c": {
            'learning_rate': 0.05,  # α
            'gamma': 0.99,  # discount factor
            'gae_lambda': 0.95,  # λ
            'ent_coef': 0.0,  # entropy coefficient
            'vf_coef': 1.0,  # value function coefficient
            'max_grad_norm': 2.0,  # max gradient norm
            'normalize_advantage': False,  # normalize advantage
  },
  "dqn": {
            'exploration_initial_eps': 1,  # ε - initial
            'exploration_final_eps': 0.15,  # ε - final
            'gradient_steps': 1,  # how many gradient steps to do after each rollout
            'target_update_interval': 32,  # update the target network every `x` steps
            'train_freq': 4  # how often the training step is done
  },
  "ppo": {
            "learning_rate": 3e-4,  # Learning rate for the optimizer
            "gamma": 0.995,  # Discount factor for future rewards
            "gae_lambda": 0.98,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            "clip_range": 0.2,  # Clipping parameter for PPO
            "ent_coef": 0.01,  # Entropy coefficient for the loss calculation
            "vf_coef": 0.5,  # Value function coefficient for the loss calculation
            "max_grad_norm": 0.5,  # Maximum value for the gradient clipping
            "policy_kwargs": {  # Additional arguments to be passed to the policy on creation
                "net_arch": [  # Custom network architecture
                    dict(pi=[128, 128], vf=[128, 128])
                ]
            }
        }
}
```

**Training** result can be found [here](/docs/training-tabulates/hyperp_noNorm_1.log). sim_time: 3_650_000 timesteps for each model.

After Welch Procedure, we have computed **output analysis** for all models (A2C, DQN, PPO and fxied cost). [Here the results](/docs/system_perf_compare/result_18062024_hyperp_noNormalization.log). sim_time = 365 * 10, seeds_number = 1000.

### 3. Using same hyperparameters, policy_kwargs added

Test without reward and observations normalizations.
Custom parameters have been used.

```json
{
  "a2c": {
            'learning_rate': 0.05,  # α
            'gamma': 0.99,  # discount factor
            'gae_lambda': 0.95,  # λ
            'ent_coef': 0.0,  # entropy coefficient
            'vf_coef': 1.0,  # value function coefficient
            'max_grad_norm': 2.0,  # max gradient norm
            'normalize_advantage': False,  # normalize advantage
            "policy_kwargs": {  # Additional arguments to be passed to the policy on creation
                "net_arch": [  # Custom network architecture
                    dict(pi=[128, 128], vf=[128, 128])
                ]
            }
  },
  "dqn": {
            'exploration_initial_eps': 1,  # ε - initial
            'exploration_final_eps': 0.15,  # ε - final
            'gradient_steps': 1,  # how many gradient steps to do after each rollout
            'target_update_interval': 32,  # update the target network every `x` steps
            'train_freq': 4 , # how often the training step is done
            "policy_kwargs": {  # Additional arguments to be passed to the policy on creation
                "net_arch": [  # Custom network architecture
                    128, 128
                ]
            }
  ,
  "ppo": {
            "learning_rate": 3e-4,  # Learning rate for the optimizer
            "gamma": 0.995,  # Discount factor for future rewards
            "gae_lambda": 0.98,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            "clip_range": 0.2,  # Clipping parameter for PPO
            "ent_coef": 0.01,  # Entropy coefficient for the loss calculation
            "vf_coef": 0.5,  # Value function coefficient for the loss calculation
            "max_grad_norm": 0.5,  # Maximum value for the gradient clipping
            "policy_kwargs": {  # Additional arguments to be passed to the policy on creation
                "net_arch": [  # Custom network architecture
                    dict(pi=[128, 128], vf=[128, 128])
                ]
            }
        }
}
```
### 4. Using same hyperparameters, batch size increased

Test without reward and observations normalizations.
Custom parameters have been used.

```json
{
  "a2c": {
            'learning_rate': 0.05,  # α
            'gamma': 0.99,  # discount factor
            'gae_lambda': 0.95,  # λ
            'ent_coef': 0.0,  # entropy coefficient
            'vf_coef': 1.0,  # value function coefficient
            'max_grad_norm': 2.0,  # max gradient norm
            'normalize_advantage': False,  # normalize advantage,
            'n_steps': 10,  # number of steps
            "policy_kwargs": {  # Additional arguments to be passed to the policy on creation
                "net_arch": [  # Custom network architecture
                    dict(pi=[128, 128], vf=[128, 128])
                ]
            }
  },
  "dqn": {
            'exploration_initial_eps': 1,  # ε - initial
            'exploration_final_eps': 0.15,  # ε - final
            'gradient_steps': 1,  # how many gradient steps to do after each rollout
            'target_update_interval': 32,  # update the target network every `x` steps
            'train_freq': 4 , # how often the training step is done
            "batch_size": 256,  # Size of the batch
            "policy_kwargs": {  # Additional arguments to be passed to the policy on creation
                "net_arch": [  # Custom network architecture
                    128, 128
                ]
            }
  ,
  "ppo": {
            "learning_rate": 3e-4,  # Learning rate for the optimizer
            "gamma": 0.995,  # Discount factor for future rewards
            "gae_lambda": 0.98,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            "clip_range": 0.2,  # Clipping parameter for PPO
            "ent_coef": 0.01,  # Entropy coefficient for the loss calculation
            "vf_coef": 0.5,  # Value function coefficient for the loss calculation
            "max_grad_norm": 0.5,  # Maximum value for the gradient clipping
            "batch_size": 256,  # Number of experiences sampled from the replay buffer for each update
            "policy_kwargs": {  # Additional arguments to be passed to the policy on creation
                "net_arch": [  # Custom network architecture
                    dict(pi=[128, 128], vf=[128, 128])
                ]
            }
        }
}
```

**Training** result can be found [here](/docs/training-tabulates/hyperp_noNorm_3.log). sim_time: 3_650_000 timesteps for each model.

After Welch Procedure, we have computed **output analysis** for all models (A2C, DQN, PPO and fxied cost). [Here the results](/docs/system_perf_compare/result_21062024_hyperp_noNormalization_3.log). sim_time = 365 * 10, seeds_number = 1000.

### Models
The following models have been trained:
- PPO
- A2C
- DQN

Also a Policy Fixed model has been implemented considering S_Min and S_Max retrivied at 2^K factorial design.

100 Seeds are employed to get the total cost within each model. Average and variance are plotted at the end of all simulations.
<br>
Monitor training executing this bash script:  tensorboard --logdir ./log/a2c_mlp_tensorboard;./log/dqn_mlp_tensorboard;./log/ppo_mlp_tensorboard

### Output
Check for the most recent picture inside [this directory](/docs/system_perf_compare)

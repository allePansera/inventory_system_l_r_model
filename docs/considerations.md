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

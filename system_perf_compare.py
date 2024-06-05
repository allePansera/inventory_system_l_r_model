"""
Runnare in modo statisticamente corretto il sistema con agent o con policy fissa.
Fare confronto printando a fine simulazione a quanto ammonta il costo totale usando la media dei valori
con la lista dei seed.

Warehouse -> sistema con gli agent
Warehouse_S -> sistema con policy fissa
"""
from system.Inventory_Multi_Item import Warehouse
from system.Inventory_Multi_Item_S import Warehouse as Warehouse_S
from system.InventoryEnv_Multi_Item import WarehouseEnv
from system.Item import Item
from agents.AgentsLoader import AgentsLoader
from tabulate import tabulate
from tqdm import tqdm
import simpy
import logging
import random
import statistics
import numpy as np


logging.getLogger('stable_baselines3').setLevel(logging.WARNING)

# Simulation time - 5 years
sim_time = 365

# Define the items
item_1 = Item(
    id="1",
    description="Iphone 15",
    lead_time=lambda: random.uniform(15, 30),
    demand_inter_arrival_time=lambda: random.expovariate(lambd=1/3),
    demand_distribution=[[1, 2, 3, 4], [1/3, 1/6, 1/6, 1/3]],
    s_min=20,
    s_max=60
)
item_2 = Item(
    id="2",
    description="AirPods Pro",
    lead_time=lambda: random.uniform(6, 21),
    demand_inter_arrival_time=lambda: random.expovariate(lambd=1/3),
    demand_distribution=[[1, 2, 3, 4], [1/3, 1/6, 1/6, 1/3]],
    s_min=20,
    s_max=60
)
items = [item_1, item_2]
# Inventory initial position
inventory_position_distribution_1 = lambda: random.uniform(-75, 75)
inventory_position_distribution_2 = lambda: random.uniform(-76.125, 76.125)

# Definizione dei seed
seeds = [42, 34, 58, 78, 11, 90, 25, 58, 90, 13]
policy_fixed_costs = []
rl_agent_costs = []
# RL agent list
print("Running RL Agent...")
for seed in tqdm(seeds):
    # Define Simpy environment
    env = simpy.Environment()
    # Define Warehouse
    w_simpy_env = Warehouse(
        id="Warehouse - Training",
        env=env,
        items=items,
        inventory_levels=[inventory_position_distribution_1, inventory_position_distribution_2],
        order_setup_cost=10,
        order_incremental_cost=3,
        holding_cost=1,
        shortage_cost=7
    )
    # Define Warehouse Gym Env
    w_gym_env = WarehouseEnv(
        warehouse=w_simpy_env,
        step_duration=1,  # 1 Day, add delay to process last cost
    )
    # Load best agent
    al = AgentsLoader(w_gym_env)
    agent = al.load_weight(model_id="a2c_mlp")
    obs, _ = w_gym_env.reset(seed=seed)
    for _ in range(sim_time):
        actions, _states = agent.predict(obs)
        obs, rewards, done, truncated, info = w_gym_env.step(actions)
    rl_agent_costs.append(w_gym_env.warehouse.total_cost)

# Policy fixed agent
print("Running policy fixed agent...")
for seed in tqdm(seeds):
    random.seed(seed)
    # Define Simpy environment
    env = simpy.Environment()
    # Define warehouse env
    w_simpy_env_S = Warehouse_S(
        id="Warehouse - Training",
        env=env,
        items=items,
        inventory_levels=[inventory_position_distribution_1, inventory_position_distribution_2],
        order_setup_cost=10,
        order_incremental_cost=3,
        holding_cost=1,
        shortage_cost=7
    )
    # Run policy fixed system for given amount of time
    w_simpy_env_S.env.run(until=sim_time)
    policy_fixed_costs.append(w_gym_env.warehouse.total_cost)

avg_policy_fixed = statistics.mean(policy_fixed_costs)
var_policy_fixed = np.var(policy_fixed_costs)
avg_rl_agent = statistics.mean(rl_agent_costs)
var_rl_agent = np.var(rl_agent_costs)

header = ["Avg. Costs 1", "Avg. Costs 2", "Var. Costs 1", "Var. Costs 2"]
print(
    tabulate(
        [
            [
                avg_policy_fixed, avg_rl_agent, var_policy_fixed, var_rl_agent
            ],
        ],
        headers=header,
        tablefmt="fancy_grid",
        floatfmt=".2f",
        numalign="center",
        stralign="center",
        colalign=("center", "center"),
        showindex=False,
        headers_style="bold"
    )
)
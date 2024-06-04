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
from utils import clean_plot_directory
from utils import clean_log_file
from tabulate import tabulate
import logging
import simpy
import random

logging_path = 'log/output.log'
# Clean plot directory
clean_plot_directory()
# Clean log
clean_log_file(logging_path)
# Simulation time - 5 years
sim_time = 365*5

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
# Define Warehouse Gym Env
w_gym_env = WarehouseEnv(
    warehouse=w_simpy_env,
    step_duration=1, # 1 Day, add delay to process last cost
)
# Define Warehouse Gym Env
w_gym_env = WarehouseEnv(
    warehouse=w_simpy_env,
    step_duration=1, # 1 Day, add delay to process last cost
)

# Load best agent
al = AgentsLoader(w_gym_env)
agent = al.load_weight(model_id="a2c_mlp")
obs, _ = w_gym_env.reset()
for _ in range(sim_time):
    actions, _states = agent.predict(obs)
    obs, rewards, done, truncated, info = w_gym_env.step(actions)
print("RL Agent total cost: ",w_gym_env.warehouse.total_cost)

# Run policy fixed system for given amount of time
# TODO


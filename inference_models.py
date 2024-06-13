from system.Inventory_Multi_Item import Warehouse
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(logging_path),
        logging.StreamHandler()
    ]
)
# Logger creation
logger = logging.getLogger(__name__)

# Define the items
item_1 = Item(
    id="1",
    description="Iphone 15",
    lead_time=lambda: random.uniform(15, 30),
    demand_inter_arrival_time=lambda: random.expovariate(lambd=1 / 3),
    demand_distribution=[[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]]
)
item_2 = Item(
    id="2",
    description="AirPods Pro",
    lead_time=lambda: random.uniform(6, 21),
    demand_inter_arrival_time=lambda: random.expovariate(lambd=1 / 3),
    demand_distribution=[[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]]
)
items = [item_1, item_2]
# Inventory initial position
inventory_position_distribution_1 = lambda: random.uniform(-75, 75)
inventory_position_distribution_2 = lambda: random.uniform(-75, 75)
# Define Simpy environment
env = simpy.Environment()
# Define Warehouse
w_simpy_env = Warehouse(
    id="Warehouse - Inference",
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

# Load all agents' model
al = AgentsLoader(w_gym_env)
al.load_weights()

# Measure the result of the simulation with RL agent
for agent in al.agents:
    obs, _ = w_gym_env.reset()
    logger.info(agent)
    headers = ["Inventory position Item 1", "Delta Quantity Ordered Item 1",
               "Delta Time Last Order Item 1", "Delta Orders Counter Item 1",
               "Order Rate Item 1",
               "Inventory position Item 2", "Delta Quantity Ordered Item 2",
               "Delta Time Last Order Item 2", "Delta Orders Counter Item 2",
               "Order Rate Item 2",
               "Action Item",
               "Reward"]

    prediction = []
    prediction.append([*obs, 0, 0])
    for _ in range(100):
        action, _state = agent.predict(obs)
        obs, rewards, done, truncated, info = w_gym_env.step(action)
        prediction.append([*obs, action, rewards])

    tab = tabulate(
        prediction,
        headers=headers,
        tablefmt="fancy_grid"
    )

# Measure the result of the simulation with policy_fixed system

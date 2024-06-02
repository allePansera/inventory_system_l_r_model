from system.Inventory import Warehouse
from system.InventoryEnv import WarehouseEnv
from agents.AgentsLoader import AgentsLoader
from utils import clean_plot_directory
from utils import clean_log_file
from tabulate import tabulate
import logging
import simpy
import random
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

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

# Lead time -> at least 5/10 days
lead_time_distribution = lambda: random.uniform(5, 10)
# Inventory initial position
inventory_position_distribution = lambda: random.uniform(-20, 100)
# Demand inter-arrival
demand_inter_arrival_mean_time_distribution = lambda: random.expovariate(lambd=1 / 0.5)
# Demand distribution
demand_distribution = [[1, 2, 3, 4], [1/3, 1/6, 1/6, 1/3]]
# Define Simpy environment
env = simpy.Environment()
# Define Warehouse
w_simpy_env = Warehouse(
    id="Warehouse - Training",
    env=env,
    lead_time=lead_time_distribution,
    demand_inter_arrival_mean_time=demand_inter_arrival_mean_time_distribution,
    demand_distribution=demand_distribution,
    inventory_level=inventory_position_distribution,
    order_setup_cost=32,
    order_incremental_cost=3,
    holding_cost=1,
    shortage_cost=5
)
# Define Warehouse Gym Env
w_gym_env = WarehouseEnv(
    warehouse=w_simpy_env,
    step_duration=1, # 1 Day, add delay to process last cost
)
# Train all agents' model
al = AgentsLoader(w_gym_env)
duration_sec = al.train(
    episode_duration=365*10, # 5 Year
    plot_rewards=True
)

logger.info(f"All agents have been trained in {duration_sec} sec")

# Run the trained agents and check for the results
for agent in al.agents:
    obs, _ = w_gym_env.reset()
    logger.info(agent)
    headers = ["Inventory position","Delta Quantity Ordered",
               "Delta Time Last Order", "Delta Orders Counter",
               "Order Rate",
               "Action", "Reward"]
    prediction = []
    prediction.append([*obs, 0, 0])
    for _ in range(20):
        action, _states = agent.predict(obs)
        obs, rewards, done, truncated, info = w_gym_env.step(action)
        prediction.append([*obs, action, rewards])

    tab = tabulate(
        prediction,
        headers=headers,
        tablefmt="fancy_grid"
    )
    print(tab)


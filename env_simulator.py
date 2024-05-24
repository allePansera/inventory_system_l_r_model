from system.Inventory import Warehouse
import simpy
import random

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
    id="Warehouse-Training",
    env=env,
    lead_time=lead_time_distribution,
    demand_inter_arrival_mean_time=demand_inter_arrival_mean_time_distribution,
    demand_distribution=demand_distribution,
    inventory_level=inventory_position_distribution,
    order_setup_cost=32,
    order_incremental_cost=3,
    holding_cost=1,
    shortage_cost=5,
    debug_mode=True
)
step_duration = 1.01
end = 0
while env.now <= 50:
    # Every day run specific merch qty
    w_simpy_env.take_action(30)
    # 30 Days running
    env.run(until=end+step_duration)
    print(w_simpy_env.system_description())
    end = env.now

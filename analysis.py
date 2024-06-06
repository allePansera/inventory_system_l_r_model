from scipy import stats

from system.Inventory_Multi_Item_S import Warehouse
from system.Item import Item
from utils import clean_plot_directory, generate_seeds
from utils import clean_log_file
from tabulate import tabulate
import statistics
import simpy
import random
import numpy as np
import logging
import matplotlib.pyplot as plt

"""
Possible combination
#  s d | sxd
0: - - | +
1: - + | -
2: + - | -
3: + + | +
"""


def plot_interaction(mean: list, s_min: int, s_max: int, d_min: int, d_max: int, title: str = "Interaction"):
    """
    Plot the interaction between the two factors. Pleane notice that combinations are considerate as follow:

    #  s d | sxd
    0: - - |  +
    1: - + |  -
    2: + - |  -
    3: + + |  +

    Therefore mean[0] is the mean of the first combination, mean[1] is the mean of the second combination and so on.
    """
    # Define the points for the first line
    x1 = [s_min, s_max]
    y1 = [mean[0], mean[2]]
    plt.plot(x1, y1, label=f'd = {d_min}')

    # Define the points for the second line
    x2 = [s_min, s_max]
    y2 = [mean[1], mean[3]]
    plt.plot(x2, y2, label=f'd = {d_max}')
    plt.title(title)
    plt.legend()
    plt.show()


def t_student_critical_value(alpha: float, n: int) -> float:
    return stats.t.ppf(1 - alpha, n - 1)


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

# Creazione di un logger
logger = logging.getLogger(__name__)

# Possibile minimum-maximum threshold
s_min_item_1 = 5
s_max_item_1 = 30
d_min_item_1 = 30
d_max_item_1 = 50

# Possibile minimum-maximum threshold
s_min_item_2 = 10
s_max_item_2 = 20
d_min_item_2 = 20
d_max_item_2 = 50

comb = {0: [(s_min_item_1, d_min_item_1), (s_min_item_1, d_max_item_1), (s_max_item_1, d_min_item_1), (s_max_item_1, d_max_item_1)],
        1: [(s_min_item_2, d_min_item_2), (s_min_item_2, d_max_item_2), (s_max_item_2, d_min_item_2), (s_max_item_2, d_max_item_2)]}

# Contains simulation of cost per month
R = {0: {}, 1: {}}
# Month of simulation
SIM_MONTH = 120
# Seeds to use for env random simulation
seeds: list[int] = generate_seeds(n=100)
# Compute, per simulation response, the mean, the variance and the half-interval
R_stats = {0: [], 1: []}
# Expected effect
E_stats = {0: [], 1: []}

for item_idx in range(2):
    for i in range(len(comb[item_idx])):
        for index, seed in enumerate(seeds):
            random.seed(seed)
            # Define the items
            item_1 = Item(
                id="1",
                description="Iphone 15",
                lead_time=lambda: random.uniform(15, 30),
                demand_inter_arrival_time=lambda: random.expovariate(lambd=1 / 3),
                demand_distribution=[[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]],
                s_min=comb[0][i][0],
                s_max=comb[0][i][0] + comb[0][i][1]
            )
            item_2 = Item(
                id="2",
                description="AirPods Pro",
                lead_time=lambda: random.uniform(6, 21),
                demand_inter_arrival_time=lambda: random.expovariate(lambd=1 / 3),
                demand_distribution=[[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]],
                s_min=comb[1][i][0],
                s_max=comb[1][i][0] + comb[1][i][1]
            )
            items = [item_1, item_2]
            # Inventory initial position
            inventory_position_distribution_1 = lambda: random.uniform(-75, 75)
            inventory_position_distribution_2 = lambda: random.uniform(-76.125, 76.125)
            distributions = [inventory_position_distribution_1, inventory_position_distribution_2]

            env = simpy.Environment()
            warehouse = Warehouse(
                id=f"item{item_idx}-{i}-{index}",
                env=env,
                items=[items[item_idx]],
                inventory_levels=[distributions[item_idx]],
                order_setup_cost=len(seeds),
                order_incremental_cost=3,
                holding_cost=1,
                shortage_cost=7,
                debug_mode=False
            )
            env.run(until=SIM_MONTH)
            if i not in R[item_idx]: R[item_idx][i] = []
            R[item_idx][i].append(statistics.mean(warehouse.daily_total_cost))
            # logger.info(warehouse.system_description(output_path=f"plot/{i}-{index}.png"))

        # Variance and Mean for each replication
        # Lecture 6.pptx
        avg_monthly_costs_sample_mean = statistics.mean(R[item_idx][i])
        avg_monthly_costs_sample_variance = statistics.variance(R[item_idx][i], xbar=avg_monthly_costs_sample_mean)
        t = t_student_critical_value(alpha=0.05, n=len(seeds))
        half_interval = t * np.sqrt(avg_monthly_costs_sample_variance / len(seeds))
        R_stats[item_idx].append({"Mean": avg_monthly_costs_sample_mean,
                                  "Variance": avg_monthly_costs_sample_variance,
                                  "Half-Interval": half_interval,
                                  "i": i, })

# Compute e_s, e_d and e_sd len(seeds) times
for item_idx in range(2):
    for i in range(len(seeds)):
        r_0, r_1, r_2, r_3 = R[item_idx][0][i], R[item_idx][1][i], R[item_idx][2][i], R[item_idx][3][i],
        e_s = (- r_0 + r_1 - r_2 + r_3) / 2
        e_d = (- r_0 - r_1 + r_2 + r_3) / 2
        e_sd = (+ r_0 - r_1 - r_2 + r_3) / 2
        E_stats[item_idx].append({"e_s": e_s,
                                  "e_d": e_d,
                                  "e_sd": e_sd,
                                  "i": i, })

# Variance and Mean for each effect replication
for item_idx in range(2):
    e_s_list = [e["e_s"] for e in E_stats[item_idx]]
    avg_e_s = statistics.mean(e_s_list)
    variance_e_s = statistics.variance(e_s_list, xbar=avg_e_s)
    t = t_student_critical_value(alpha=0.05, n=len(seeds))
    half_interval_s = t * np.sqrt(variance_e_s / len(seeds))

    e_d_list = [e["e_d"] for e in E_stats[item_idx]]
    avg_e_d = statistics.mean(e_d_list)
    variance_e_d = statistics.variance(e_d_list, xbar=avg_e_d)
    t = t_student_critical_value(alpha=0.05, n=len(seeds))
    half_interval_d = t * np.sqrt(variance_e_d / len(seeds))

    e_sd_list = [e["e_sd"] for e in E_stats[item_idx]]
    avg_e_sd = statistics.mean(e_sd_list)
    variance_e_sd = statistics.variance(e_sd_list, xbar=avg_e_sd)
    t = t_student_critical_value(alpha=0.05, n=len(seeds))
    half_interval_sd = t * np.sqrt(variance_e_sd / len(seeds))

    table = tabulate(
        [[i, R_stats[item_idx][i]["Mean"], R_stats[item_idx][i]["Variance"], R_stats[item_idx][i]["Half-Interval"]] for i in range(4)],
        ["Comb. #", "Mean", "Variance", "Half-Interval"],
        tablefmt='pretty')
    logger.info("\n" + table)

    table = tabulate(
        [
            ["E(e_s)", avg_e_sd, half_interval_s],
            ["E(e_d)", avg_e_d, half_interval_d],
            ["E(e_sd)", avg_e_sd, half_interval_sd]
        ],
        ["Comb. #", "Mean", "Half-Interval"],
        tablefmt='pretty')
    logger.info("\n" + table)
    if item_idx == 0:
        plot_interaction([R_stats[item_idx][i]["Mean"] for i in range(4)], s_min_item_1, s_max_item_1, d_min_item_1, d_max_item_1, title="Interaction Item 1")
    elif item_idx == 1:
        plot_interaction([R_stats[item_idx][i]["Mean"] for i in range(4)], s_min_item_2, s_max_item_2, d_min_item_2, d_max_item_2, title="Interaction Item 2")
    else:
        raise ValueError("Item index not valid")

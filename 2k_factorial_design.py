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
from tqdm import tqdm

"""
Possible combination
#  s d | sxd
0: - - | +
1: - + | -
2: + - | -
3: + + | +
"""


def plot_interaction(mean: list, s_min: int, s_max: int, d_min: int, d_max: int, title: str = "Interaction", ax=None):
    """
    Plot the interaction between the two factors. Please notice that combinations are considered as follows:

    #  s d | sxd
    0: - - |  +
    1: - + |  -
    2: + - |  -
    3: + + |  +

    Therefore mean[0] is the mean of the first combination, mean[1] is the mean of the second combination and so on.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Define the points for the first line
    x1 = [s_min, s_max]
    y1 = [mean[0], mean[2]]
    ax.plot(x1, y1, label=f'd = {d_min}', linewidth=2.0)

    # Define the points for the second line
    x2 = [s_min, s_max]
    y2 = [mean[1], mean[3]]
    ax.plot(x2, y2, label=f'd = {d_max}', linewidth=2.0)

    # Add a label over the line at a specific point
    ax.text(x1[0] + 0.2, y1[0] + 0.3, f'd = {d_min}', fontsize=16, ha='center')
    ax.text(x2[0] + 0.2, y2[0] + 0.3, f'd = {d_max}', fontsize=16, ha='center')

    # Add axis labels
    ax.set_xlabel('s', fontsize=16)
    ax.set_ylabel('Mean', fontsize=16)

    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=14)

    return ax


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


## VALORI PER ITEM1 OTTIMO
# s_min_range = [i for i in range(10, 15)]
# s_max_range = [i for i in range(25, 30)]
# d_min_range = [i for i in range(20, 25)]
# d_max_range = [i for i in range(30, 35)]

## VALORI PER ITEM2 OTTIMO
s_min_range = [i for i in range(3, 8)]
s_max_range = [i for i in range(8, 13)]
d_min_range = [i for i in range(20, 25)]
d_max_range = [i for i in range(30, 35)]

combinations = {i: [(s_min, d_min), (s_min, d_max), (s_max, d_min), (s_max, d_max)]
                for i, (s_min, s_max, d_min, d_max) in enumerate(zip(s_min_range, s_max_range, d_min_range, d_max_range))}

# Contains simulation of cost per month
R = {0: {}, 1: {}}
# Month of simulation
SIM_MONTH = 1_000
# Seeds to use for env random simulation
seeds: list[int] = generate_seeds(n=100)
# Compute, per simulation response, the mean, the variance and the half-interval
R_stats = {0: {}, 1: {}}
# Expected effect
E_stats = {0: {}, 1: {}}

# Select an item
for item_idx in range(2):
    # Select a list of combinations: [(-,-), (-,+), (+,-), (+,+)]
    print(f"\n\nItem {item_idx}")
    for comb_key, comb_list in combinations.items():
        print(f"\nCombination: {comb_key} - {comb_list}")
        # Select a combination (s, d)
        for comb in comb_list:
            # Compute n=10 analysis for each combination
            for index in tqdm(range(len(seeds))):
                seed = seeds[index]
                random.seed(seed)
                # Define the items
                item_1 = Item(
                    id="1",
                    description="Iphone 15",
                    lead_time=lambda: random.uniform(15, 30),
                    demand_inter_arrival_time=lambda: random.expovariate(lambd=1 / 3),
                    demand_distribution=[[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]],
                    s_min=comb[0],
                    s_max=comb[0] + comb[1]
                )
                item_2 = Item(
                    id="2",
                    description="AirPods Pro",
                    lead_time=lambda: random.uniform(6, 21),
                    demand_inter_arrival_time=lambda: random.expovariate(lambd=1 / 3),
                    demand_distribution=[[1, 2, 3, 4], [1 / 3, 1 / 6, 1 / 6, 1 / 3]],
                    s_min=comb[0],
                    s_max=comb[0] + comb[1]
                )
                items = [item_1, item_2]
                # Inventory initial position

                env = simpy.Environment()
                warehouse = Warehouse(
                    id=f"item{item_idx}-{comb}-{index}",
                    env=env,
                    items=[items[item_idx]],
                    inventory_levels=[lambda: random.uniform(-75, 75)],
                    order_setup_cost=10,
                    order_incremental_cost=3,
                    holding_cost=1,
                    shortage_cost=7
                )
                env.run(until=SIM_MONTH)
                if comb_key not in R[item_idx]: R[item_idx][comb_key] = {}
                if comb not in R[item_idx][comb_key]: R[item_idx][comb_key][comb] = []
                R[item_idx][comb_key][comb].append(statistics.mean(warehouse.daily_total_cost))
                # if index == 0: logger.info(warehouse.system_description(output_path=f"plot/{item_idx}-{comb_key}-{comb}.png"))

            # Variance and Mean for each replication
            # Lecture 6.pptx
            avg_monthly_costs_sample_mean = statistics.mean(R[item_idx][comb_key][comb])
            avg_monthly_costs_sample_variance = statistics.variance(R[item_idx][comb_key][comb], xbar=avg_monthly_costs_sample_mean)
            t = t_student_critical_value(alpha=0.05, n=len(seeds))
            half_interval = t * np.sqrt(avg_monthly_costs_sample_variance / len(seeds))
            if comb_key not in R_stats[item_idx]: R_stats[item_idx][comb_key] = {}
            if comb not in R_stats[item_idx][comb_key]: R_stats[item_idx][comb_key][comb] = []
            R_stats[item_idx][comb_key][comb] = {"Mean": avg_monthly_costs_sample_mean,
                                                 "Variance": avg_monthly_costs_sample_variance,
                                                 "Half-Interval": half_interval,
                                                 "comb": comb, }

# Compute e_s, e_d and e_sd len(seeds) times
for item_idx in range(2):
    for comb_key, comb_list in combinations.items():
        R0 = R[item_idx][comb_key][comb_list[0]]  # (-,-) -> [...]
        R1 = R[item_idx][comb_key][comb_list[1]]  # (-,+) -> [...]
        R2 = R[item_idx][comb_key][comb_list[2]]  # (+,-) -> [...]
        R3 = R[item_idx][comb_key][comb_list[3]]  # (+,+) -> [...]
        if comb_key not in E_stats[item_idx]: E_stats[item_idx][comb_key] = []
        for i in range(len(seeds)):
            r_0, r_1, r_2, r_3 = R0[i], R1[i], R2[i], R3[i]
            e_s = (- r_0 + r_1 - r_2 + r_3) / 2
            e_d = (- r_0 - r_1 + r_2 + r_3) / 2
            e_sd = (+ r_0 - r_1 - r_2 + r_3) / 2
            E_stats[item_idx][comb_key].append({"e_s": e_s,
                                                "e_d": e_d,
                                                "e_sd": e_sd,
                                                "i": i, })

# Variance and Mean for each effect replication
fig, axs = plt.subplots(nrows=2, ncols=len(combinations), figsize=(10 * len(combinations), 25))  # adjust as needed
for item_idx in range(2):
    print(f"Item {item_idx}")
    for comb_key, comb_list in combinations.items():
        print(f"Combination: {comb_key} - {comb_list}")
        e_s_list = [e["e_s"] for e in E_stats[item_idx][comb_key]]
        avg_e_s = statistics.mean(e_s_list)
        variance_e_s = statistics.variance(e_s_list, xbar=avg_e_s)
        t = t_student_critical_value(alpha=0.05, n=len(seeds))
        half_interval_s = t * np.sqrt(variance_e_s / len(seeds))

        e_d_list = [e["e_d"] for e in E_stats[item_idx][comb_key]]
        avg_e_d = statistics.mean(e_d_list)
        variance_e_d = statistics.variance(e_d_list, xbar=avg_e_d)
        t = t_student_critical_value(alpha=0.05, n=len(seeds))
        half_interval_d = t * np.sqrt(variance_e_d / len(seeds))

        e_sd_list = [e["e_sd"] for e in E_stats[item_idx][comb_key]]
        avg_e_sd = statistics.mean(e_sd_list)
        variance_e_sd = statistics.variance(e_sd_list, xbar=avg_e_sd)
        t = t_student_critical_value(alpha=0.05, n=len(seeds))
        half_interval_sd = t * np.sqrt(variance_e_sd / len(seeds))

        table = tabulate(
            [[i, R_stats[item_idx][comb_key][comb_list[i]]["Mean"], R_stats[item_idx][comb_key][comb_list[i]]["Variance"], R_stats[item_idx][comb_key][comb_list[i]]["Half-Interval"]] for i in
             range(4)],
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

        plot_interaction([R_stats[item_idx][comb_key][comb_list[i]]["Mean"] for i in range(4)],
                         s_min_range[comb_key], s_max_range[comb_key], d_min_range[comb_key], d_max_range[comb_key],
                         title=f"Interaction Item {item_idx} - {(s_min_range[comb_key], s_max_range[comb_key])}x{(d_min_range[comb_key], d_max_range[comb_key])}",
                         ax=axs[item_idx][comb_key])

plt.tight_layout(w_pad=10)
plt.show()

"""Example:
Recap of inventory system current execution:
Total cost (€): 13901
"""

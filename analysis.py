from scipy import stats


def t_student_critical_value(alpha: float, n: int) -> float:
    return stats.t.ppf(1 - alpha, n - 1)


"""
from Inventory import Warehouse
from analysis import t_student_critical_value
from utils import clean_plot_directory
from utils import clean_log_file
from tabulate import tabulate
import statistics
import simpy
import random
import numpy as np
import logging

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
s_comb = [(20, 30), (20, 70), (60, 70), (60, 110)]
# Contains simulation of cost per month
R = {}
# Month of simulation
SIM_MONTH = 120
# Seeds to use for env random simulation
seeds = [42, 34, 58, 78, 11, 90, 25, 58, 90, 13]
# Compute, per simulation response, the mean, the variance and the half-interval
R_stats = []
# Expected effect
E_stats = []

# Lead time
lead_time_distribution = lambda: random.uniform(0.5, 1)
# Demand inter-arrival
demand_inter_arrival_mean_time_distribution = lambda: random.expovariate(lambd=1 / 0.1)
# Demand distribution
demand_distribution = [[1, 2, 3, 4], [1/3, 1/6, 1/6, 1/3]]


for i in range(len(s_comb)):
    for index, seed in enumerate(seeds):
        random.seed(seed)
        env = simpy.Environment()
        warehouse = Warehouse(
            id=f"{i}-{index}",
            env=env,
            lead_time=lead_time_distribution,
            demand_inter_arrival_mean_time=demand_inter_arrival_mean_time_distribution,
            demand_distribution=demand_distribution,
            inventory_level=60,
            inventory_check_interval=1,
            s_max=s_comb[i][1],
            s_min=s_comb[i][0],
            order_setup_cost=32,
            order_incremental_cost=3,
            holding_cost=1,
            shortage_cost=5
        )
        env.run(until=SIM_MONTH)
        if i not in R: R[i] = []
        R[i].append(statistics.mean(warehouse.month_total_cost))
        logger.info(warehouse.system_description(output_path=f"plot/{i}-{index}.png"))

    # Variance and Mean for each replication
    # Lecture 6.pptx
    avg_monthly_costs_sample_mean = statistics.mean(R[i])
    avg_monthly_costs_sample_variance = statistics.variance(R[i], xbar=avg_monthly_costs_sample_mean)
    t = t_student_critical_value(alpha=0.05, n=10)
    half_interval = t * np.sqrt(avg_monthly_costs_sample_variance / 10)
    R_stats.append({"Mean": avg_monthly_costs_sample_mean,
                    "Variance": avg_monthly_costs_sample_variance,
                    "Half-Interval": half_interval,
                    "i": i, })

# Compute e_s, e_d and e_sd 10 times
for i in range(10):
    r_0, r_1, r_2, r_3 = R[0][i], R[1][i], R[2][i], R[3][i],
    e_s = (- r_0 + r_1 - r_2 + r_3)/2
    e_d = (- r_0 - r_1 + r_2 + r_3)/2
    e_sd =(+ r_0 - r_1 - r_2 + r_3)/2
    E_stats.append({"e_s": e_s,
                    "e_d": e_d,
                    "e_sd": e_sd,
                    "i": i, })

# Variance and Mean for each effect replication
e_s_list = [ e["e_s"] for e in E_stats ]
avg_e_s = statistics.mean(e_s_list)
variance_e_s = statistics.variance(e_s_list, xbar=avg_e_s)
t = t_student_critical_value(alpha=0.05, n=10)
half_interval_s = t * np.sqrt(variance_e_s/ 10)

e_d_list = [ e["e_d"] for e in E_stats ]
avg_e_d = statistics.mean(e_d_list)
variance_e_d = statistics.variance(e_d_list, xbar=avg_e_d)
t = t_student_critical_value(alpha=0.05, n=10)
half_interval_d = t * np.sqrt(variance_e_d/ 10)


e_sd_list = [ e["e_sd"] for e in E_stats ]
avg_e_sd = statistics.mean(e_sd_list)
variance_e_sd = statistics.variance(e_sd_list, xbar=avg_e_sd)
t = t_student_critical_value(alpha=0.05, n=10)
half_interval_sd = t * np.sqrt(variance_e_sd/ 10)

table = tabulate(
        [[i, R_stats[i]["Mean"], R_stats[i]["Variance"], R_stats[i]["Half-Interval"]] for i in range(4)],
        ["Comb. #", "Mean", "Variance", "Half-Interval"],
        tablefmt='pretty')
logger.info("\n" + table)

table = tabulate(
        [
            ["E(e_s)", avg_e_sd, half_interval_s] ,
            ["E(e_d)", avg_e_d, half_interval_d],
            ["E(e_sd)", avg_e_sd, half_interval_sd]
        ],
        ["Comb. #", "Mean", "Half-Interval"],
        tablefmt='pretty')
logger.info("\n" + table)


"""

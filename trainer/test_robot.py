"""Aggregate eval.log files of 3 seeds into the paper table (mean / std over seeds at step 250000).

Usage (run from trainer/):
    python test_robot.py --environment robot_reach                    # DART (mixing_ratio 0.7, xy)
    python test_robot.py --environment robot_pegbox                   # DART (mixing_ratio 0.4, xyz)
    python test_robot.py --environment robot_reach  --method baseline # vanilla SAC (batch 64)
    python test_robot.py --environment robot_pegbox --method baseline

Writes eval_<environment>.csv (DART) or eval_<environment>_baseline.csv (baseline).
Columns: Test Env 1..5 = int(mean over seeds), "bound" = int(std over seeds).
"""
import pandas as pd
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--environment', required=True, choices=["robot_reach", "robot_pegbox"])
parser.add_argument('--method', default="dart", choices=["dart", "baseline"])
parser.add_argument('--seeds', default="10081,10082,10083",
                    help="comma-separated seeds (paper: 10081,10082,10083)")
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]

# Run-directory tags written by train.py (see make_run_tag): '.' is replaced by 'p'.
if args.environment == "robot_reach":
    action_space, mr = "xy", "0p7"
else:
    action_space, mr = "xyz", "0p4"

if args.method == "dart":
    group_1 = [f"seed{s}_ts250000_bs32_proj256_auxlr0p001_as{action_space}_fs3_k1_mr{mr}_tmax100/esdda"
               for s in seeds]
else:
    group_1 = [f"seed{s}_ts250000_bs64_proj256_auxlr0p001_as{action_space}_fs3" for s in seeds]
group_1 = [os.path.join("./logs", args.environment, "sac", g) for g in group_1]

# column names match the eval.log keys
combined_data = pd.DataFrame(columns=[
    "Algorithm", "Step", "Episode Reward",
    "Test Env 1", "Test Env 2", "Test Env 3", "Test Env 4", "Test Env 5"
])
algorithm_data = []

group1_name = "sac"

algo_lst = [group1_name]
algo_path_lst = [group_1]

for algo, algo_path in zip(algo_lst, algo_path_lst):
    element = {}
    element["environment"] = args.environment
    element["algorithm"] = algo
    element["files"] = []

    for file in algo_path:
        element["files"].append(os.path.join(file, "eval.log"))

    algorithm_data.append(element)

for algo_info in algorithm_data:
    algo = algo_info["algorithm"]
    for log_file in algo_info["files"]:
        with open(log_file, 'r') as file:
            data = [eval(line) for line in file]
            has_100k_step = False
            for d in data:
                if d["step"] == 100000:
                    has_100k_step = True
                df = pd.DataFrame(data=[[
                    algo, d["step"], d["episode_reward"],
                    d["episode_reward_test_env_1"],
                    d["episode_reward_test_env_2"],
                    d["episode_reward_test_env_3"],
                    d["episode_reward_test_env_4"],
                    d["episode_reward_test_env_5"]
                ]],
                columns=[
                    "Algorithm", "Step", "Episode Reward",
                    "Test Env 1", "Test Env 2", "Test Env 3", "Test Env 4", "Test Env 5"
                ])
                combined_data = pd.concat([combined_data, df])

            if not has_100k_step:
                raise ValueError(f"Step 100,000 not found in file: {log_file}")


eval_csv = pd.DataFrame(columns=[
    "Algorithm", "Test Env 1", "bound",
    "Test Env 2", "bound", "Test Env 3", "bound",
    "Test Env 4", "bound", "Test Env 5", "bound"
])

for algo in algo_lst:
    # keep only the final evaluation (step 250000)
    filtered_df = combined_data[(combined_data['Algorithm'] == algo) & (combined_data['Step'] == 250000)]
    print(filtered_df)

    envs = ["Test Env 1", "Test Env 2", "Test Env 3", "Test Env 4", "Test Env 5"]
    row = [algo]
    for env in envs:
        row.append(int(np.mean(filtered_df[env])))
        row.append(int(np.std(filtered_df[env])))

    df = pd.DataFrame(data=[row],
                      columns=["Algorithm",
                               "Test Env 1", "bound",
                               "Test Env 2", "bound",
                               "Test Env 3", "bound",
                               "Test Env 4", "bound",
                               "Test Env 5", "bound"])
    eval_csv = pd.concat([eval_csv, df])

suffix = "" if args.method == "dart" else "_baseline"
out = "eval_{}{}.csv".format(args.environment, suffix)
eval_csv.to_csv(out)
print("wrote", out)

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import argparse
import numpy as np


parser = argparse.ArgumentParser()

# environment
parser.add_argument('--environment')

args = parser.parse_args()

combined_data = pd.DataFrame(columns=["Algorithm", "Step", "Episode Reward","Color Easy"
    ,"Color Hard","Video Easy","Video Hard"])
algorithm_data=[]


group_2=["1111_100_0.1_32_1.0","2222_100_0.1_32_1.0","7824_100_0.1_32_1.0"]
group_3=["1111_100_0.5_32_1.0","2222_100_0.5_32_1.0","7824_100_0.5_32_1.0"]
group_4=["1111_100_0.2_32_1.0","2222_100_0.2_32_1.0","7824_100_0.2_32_1.0"]
group_5=["1111_100_0.5_32_1.0","2222_100_0.5_32_1.0","7824_100_0.5_32_1.0"]
group_6=["1111_100_0.2_32_1.0","2222_100_0.2_32_1.0","7824_100_0.2_32_1.0"]





group_2=[os.path.join("logs","walker_walk","sac",g) for g in group_2]
group_3=[os.path.join("logs","walker_stand","sac",g) for g in group_3]
group_4=[os.path.join("logs","cartpole_swingup","sac",g) for g in group_4]
group_5=[os.path.join("logs","ball_in_cup_catch","sac",g) for g in group_5]
group_6=[os.path.join("logs","finger_spin","sac",g) for g in group_6]


group2_name="walker_walk"
group3_name="walker_stand"
group4_name="cartpole_swingup"
group5_name="ball_in_cup_catch"
group6_name="finger_spin"




algo_lst=[group2_name,group3_name,
          group4_name,group5_name,group6_name]
algo_path_lst=[group_2,group_3,
               group_4,group_5,group_6]

for algo,algo_path in zip(algo_lst,algo_path_lst):
    element={}
    element["environment"]=args.environment
    element["algorithm"]=algo
    element["files"]=[]

    for file in algo_path:
        element["files"].append(os.path.join(file,"eval.log"))

    algorithm_data.append(element)

# Create the plot
plt.figure(figsize=(10, 6))

for algo_info in algorithm_data:
    algo = algo_info["algorithm"]
    for log_file in algo_info["files"]:
        with open(log_file, 'r') as file:
            data = [eval(line) for line in file]
            has_final_step = False  # whether a step==100000 record exists
            for d in data:
                if d["step"] == 100000:
                    has_final_step = True
                df = pd.DataFrame(data=[[algo, d["step"], d["episode_reward"], d["episode_reward_color_easy"],
                                         d["episode_reward_color_hard"], d["episode_reward_video_easy"], d["episode_reward_video_hard"]]],
                                  columns=["Algorithm", "Step", "Episode Reward", "Color Easy",
                                           "Color Hard", "Video Easy", "Video Hard"])
                combined_data = pd.concat([combined_data, df])
            
            if not has_final_step:
                raise ValueError(f"Step 100,000 not found in file: {log_file}")


eval_csv = pd.DataFrame(columns=["Algorithm","Color Easy","bound",
                                 "Color Hard","bound","Video Easy","bound","Video Hard","bound"])

for algo in algo_lst:
    # Keep only the step==100000 rows of this environment
    filtered_df = combined_data[(combined_data['Algorithm'] == algo) & (combined_data['Step'] == 100000)]
    print(filtered_df)
    # Pull out the four evaluation columns
    color_easy = filtered_df['Color Easy']
    color_hard = filtered_df['Color Hard']
    video_easy = filtered_df['Video Easy']
    video_hard = filtered_df['Video Hard']
    # Mean / std over seeds
    average_color_easy = np.mean(color_easy)
    average_color_hard = np.mean(color_hard)
    average_video_easy = np.mean(video_easy)
    average_video_hard = np.mean(video_hard)
    data=[algo,int(average_color_easy),int(np.std(color_easy)),
          int(average_color_hard),int(np.std(color_hard)),
          int(average_video_easy),int(np.std(video_easy)),
          int(average_video_hard),int(np.std(video_hard))]
    df = pd.DataFrame(data=[data],
                      columns=["Algorithm","Color Easy","bound",
                                 "Color Hard","bound","Video Easy","bound","Video Hard","bound"])
    eval_csv = pd.concat([eval_csv, df])

    everything=pd.concat([filtered_df['Color Easy'],filtered_df['Color Hard'],filtered_df['Video Easy'],filtered_df['Video Hard']])
eval_csv.to_csv("eval_{}.csv".format(args.environment))

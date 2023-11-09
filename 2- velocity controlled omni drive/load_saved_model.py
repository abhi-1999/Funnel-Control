import Robot_env
import numpy as np
import gymnasium as gym
import matplotlib as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
# import Robot_env

Use_torch = int(input("want to use torch (0-no , 1-yes): "))
if Use_torch == 0:
    register(
        # unique identifier for the env `name-version`
        id="RobotEnv-v0",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="Robot_env:RobotEnv",
    )
else:
    register(
        # unique identifier for the env `name-version`
        id="RobotEnv-v0",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="Robot_env_torch:RobotEnv",
    )

algorithm_name = input("algorithm name: ")
EpiLen = input("Episode Length: ")

env = make_vec_env("RobotEnv-v0", n_envs=12, seed=0, env_kwargs={"epi_len": int(EpiLen)})
model_path = "/home/abhijeet/Funnel-Control/2- velocity controlled omni drive/pre_saved_models/SAC/best_model_SAC_600EpiLen"
if algorithm_name=="PPO":
    model = PPO.load(model_path, env=env)
elif algorithm_name=="SAC":
    model = SAC.load(model_path, env=env)

best=-1000
for i in range(100):
    model.learn(total_timesteps=int(EpiLen)*1000)
    temp = evaluate_policy(model,model.env,n_eval_episodes=5)
    if (temp[0]>best):
        # print("best")
        best=temp[0]
        model.save("./best_models/"+algorithm_name+"/best_model_"+algorithm_name+"_"+EpiLen+"EpiLen")
    if (i%10==0):
        model.save("./models/"+algorithm_name+"/model_"+algorithm_name+"_"+EpiLen+"EpiLen")

model.save("./models/"+algorithm_name+"/model_"+algorithm_name+"_"+EpiLen+"EpiLen")

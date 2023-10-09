import Robot_env
import numpy as np
import gymnasium as gym
import matplotlib as plt
from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
# import Robot_env

register(
    # unique identifier for the env `name-version`
    id="RobotEnv-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="Robot_env:RobotEnv",
)
# Instantiate the env
#vec_env = make_vec_env(Robot_env.RobotEnv, n_envs=1)
# env = gym.make("RobotEnv-v0")
env = make_vec_env("RobotEnv-v0", n_envs=2, seed=0)

algorithm_name = input("algorithm name: ")
EpiLen = input("Episode Length: ")
if algorithm_name=="PPO":
    model = PPO("MlpPolicy",env,verbose=1,tensorboard_log="./tensorboard/"+algorithm_name+"/tensorboard_"+algorithm_name+"_"+EpiLen+"EpiLen/")
elif algorithm_name=="SAC":
    model = SAC("MlpPolicy",env,verbose=1,tensorboard_log="./tensorboard/"+algorithm_name+"/tensorboard_"+algorithm_name+"_"+EpiLen+"EpiLen/")
model.learn(total_timesteps=int(EpiLen)*100000)
model.save("./models/"+algorithm_name+"/model_"+algorithm_name+"_"+EpiLen+"EpiLen")
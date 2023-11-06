import Robot_env
import numpy as np
import gymnasium as gym
import matplotlib as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
import time
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

env = make_vec_env("RobotEnv-v0", n_envs=7, seed=0, env_kwargs={"epi_len": int(EpiLen)})
#env = gym.make("RobotEnv-v0", env_kwargs={"epi_len": EpiLen})
if algorithm_name=="PPO":
    model = PPO("MlpPolicy",env,verbose=1,tensorboard_log="./tensorboard/"+algorithm_name+"/tensorboard_"+algorithm_name+"_"+EpiLen+"EpiLen/")
elif algorithm_name=="SAC":
    model = SAC("MlpPolicy",env,verbose=1,tensorboard_log="./tensorboard/"+algorithm_name+"/tensorboard_"+algorithm_name+"_"+EpiLen+"EpiLen/")
best=0
for i in range(100):
    # start = time.time()
    model.learn(total_timesteps=int(EpiLen)*10000)
    # end = time.time()
    # print((end - start)* 10**3, "ms")
    temp = evaluate_policy(model,model.env,n_eval_episodes=5)
    if (temp[0]>best):
        # print("best")
        best=temp[0]
        model.save("./best_models/"+algorithm_name+"/best_model_"+algorithm_name+"_"+EpiLen+"EpiLen")
    if (i%10==0):
        model.save("./models/"+algorithm_name+"/model_"+algorithm_name+"_"+EpiLen+"EpiLen")

model.save("./models/"+algorithm_name+"/model_"+algorithm_name+"_"+EpiLen+"EpiLen")

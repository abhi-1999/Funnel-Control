import Robot_env
import numpy as np
import gymnasium as gym
import matplotlib as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
import Robot_env

register(
    # unique identifier for the env `name-version`
    id="RobotEnv-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="Robot_env:RobotEnv",
)
# Instantiate the env
#vec_env = make_vec_env(Robot_env.RobotEnv, n_envs=1)
env = gym.make("RobotEnv-v0")
model = PPO("MlpPolicy",env,verbose=1)

best = -np.inf


plot_reward = []  

for i in range(400):
  model.learn(total_timesteps=2500)
  temp = evaluate_policy(model,model.env,n_eval_episodes=5)
  print("reward", temp, flush = True)
  plot_reward.append(temp)
  if(temp[0]>best):
    model.save("./model_best/omnirobot")
    best = temp[0]
  if(i%10==0):
    model.save("./model/omnirobot")

np.save('omnirobot.npy',np.array(plot_reward))
model.save("./model/omnirobot")

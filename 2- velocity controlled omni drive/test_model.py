from stable_baselines3 import HerReplayBuffer, PPO, SAC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register

 

Use_torch = int(input("want to use torch (0-no , 1-yes): "))
algorithm_name = input("algorithm name: ")
EpiLen = input("Episode Length: ")
best_model = int(input("want to use best_model(0-no 1-yes): "))
 

if Use_torch == 0:
    register(
        # unique identifier for the env `name-version`
        id="RobotEnv-v0",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="Robot_env:RobotEnv",
        kwargs={'epi_len': int(EpiLen)},
    )
else:
    register(
        # unique identifier for the env `name-version`
        id="RobotEnv-v0",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="Robot_env_torch:RobotEnv",
        kwargs={'epi_len': int(EpiLen)},
    )
env = gym.make("RobotEnv-v0", env_kwargs={"epi_len": EpiLen})
# model_folder_name = "./model/"+"model_"+algorithm_name+"_"+EpiLen+"EpiLen"
if best_model == 0:
    model_path = "./models/"+algorithm_name+"/model_"+algorithm_name+"_"+EpiLen+"EpiLen"
else :
    model_path = "./best_models/"+algorithm_name+"/best_model_"+algorithm_name+"_"+EpiLen+"EpiLen"

if algorithm_name=="PPO":
    model = PPO.load(model_path, env=env)
elif algorithm_name=="SAC":
    model = SAC.load(model_path, env=env)

 

x_min,x_max,y_min,y_max = [],[],[],[]
x_state,y_state,theta_state,rew = [],[],[],[]
for _ in range(1):
    obs, info = env.reset()
    print(obs)
    for i in range(int(EpiLen)):
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action,env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        x_state.append(obs[0])
        y_state.append(obs[1])
        theta_state.append(obs[2])
        rew.append(reward)
        x_min.append(info['x_min'])
        x_max.append(info['x_max'])
        y_min.append(info['y_min'])
        y_max.append(info['y_max'])

        if terminated or truncated:
            obs, info = env.reset()
            break
    env.render()
env.close()

time = np.linspace(0,int(EpiLen),int(EpiLen))
plt.subplot(221)
plt.plot(time,x_min,time,x_max,time,x_state)
plt.xlabel("Time",fontsize=15)
plt.ylabel('x - state',fontsize=15)
plt.legend(['x_lb','x_ub','x_act'])
plt.grid()

plt.subplot(222)
plt.plot(time,y_min,time,y_max,time,y_state)
plt.legend(['y_lb','y_ub','y_act'])
plt.xlabel("Time",fontsize=15)
plt.ylabel('y - state',fontsize=15)
plt.grid()
plt.show()

time = np.linspace(0,5,int(EpiLen))
time_int = 0.01
xd=[[1.3+0.4*np.cos(1.9*time_int*t+1.5), 3.5+0.6*np.sin(1.9*t*time_int+1.5)] for t in range(int(EpiLen))]
state_d = np.array(xd)
x_lb_hard = [1 for i in range(int(EpiLen))]
x_ub_hard = [2 for i in range(int(EpiLen))]
lb_hard_y = [3 for i in range(int(EpiLen))]
ub_hard_y = [4 for i in range(int(EpiLen))]
plt.plot(state_d[:,0],state_d[:,1],x_lb_hard,time,x_ub_hard,time,time,lb_hard_y,time,ub_hard_y,x_state,y_state)
plt.grid()
plt.show()
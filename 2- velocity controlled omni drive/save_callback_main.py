import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
# import Robot_env

Use_torch = int(input("want to use torch (0-no , 1-yes): "))
if Use_torch == 0:
    register(
        id="RobotEnv-v0",
        entry_point="Robot_env:RobotEnv",
    )
else:
    register(
        id="RobotEnv-v0",
        entry_point="Robot_env_torch:RobotEnv",
    )

algorithm_name = input("algorithm name: ")
EpiLen = input("Episode Length: ")

env = make_vec_env("RobotEnv-v0", n_envs=2, seed=0, env_kwargs={"epi_len": int(EpiLen)})



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1,EpiLen=EpiLen):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_models/'+algorithm_name+'/model_'+algorithm_name+'_'+EpiLen+"EpiLen")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
if algorithm_name=="PPO":
    model = PPO("MlpPolicy",env,verbose=1,tensorboard_log="./tensorboard/"+algorithm_name+"/tensorboard_"+algorithm_name+"_"+EpiLen+"EpiLen/")
elif algorithm_name=="SAC":
    model = SAC("MlpPolicy",env,verbose=1,tensorboard_log="./tensorboard/"+algorithm_name+"/tensorboard_"+algorithm_name+"_"+EpiLen+"EpiLen/")
best=0
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir,verbose=1,EpiLen=EpiLen)

model.learn(total_timesteps=int(EpiLen)*1000000, callback = callback)
model.save("./models/"+algorithm_name+"/model_"+algorithm_name+"_"+EpiLen+"EpiLen")

time_steps = int(EpiLen)*1000000

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, algorithm_name)
plt.show()
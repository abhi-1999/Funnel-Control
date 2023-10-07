from stable_baselines3.common.env_checker import check_env
import Robot_env_tensor
env = Robot_env_tensor.RobotEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)
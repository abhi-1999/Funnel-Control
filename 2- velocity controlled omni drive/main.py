from stable_baselines3.common.env_checker import check_env
import Robot_env
env = Robot_env.RobotEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)
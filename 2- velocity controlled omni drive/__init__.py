from gymnasium.envs.registration import register
import Robot_env

register(
    # unique identifier for the env `name-version`
    id="RobotEnv-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="Robot_env:RobotEnv",
)
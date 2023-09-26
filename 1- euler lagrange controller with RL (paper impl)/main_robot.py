import numpy as np
import torch
# import gym
import argparse
import os

import utils
import TD3
import RobotEnv
import config
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = RobotEnv.RobotEnv()
 	# eval_env = RobotEnv.RobotEnv() #gym.make(env_name)
	# eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

if True:#__name__ == "__main__":
	print("Program started")
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="RobotEnv")
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e7, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=5, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")
	if not os.path.exists("./models_best"):
		os.makedirs("./models_best")

	# env = gym.make(args.env)
	env = RobotEnv.RobotEnv()
	# Set seeds
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.get_state_dim()
	action_dim = env.get_action_dim()
	max_action = env.max_action
	_max_episode_steps = env.epi_len
 
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}
	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = [args.policy_noise * i for i in max_action]# args.policy_noise * max_action
		kwargs["noise_clip"] = [args.noise_clip * i for i in max_action]
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	best = -500
	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		# Select action randomly or according to policy
		# if True:
		if t < args.start_timesteps:
			action = env.sample()
		else:
			action_range = torch.mul(torch.Tensor(max_action),args.expl_noise)
			# print("action_range:",action_range)
			# print(type(policy.select_action(np.array(state))
			# 	+ np.random.normal(0, action_range, size=action_dim)))
   
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, action_range, size=action_dim)
			).clip(-np.array(max_action), np.array(max_action))

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < _max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t <= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			if (episode_reward>best):
				policy.save(f"./models_best/{file_name}")
				best = episode_reward
				print("best: ", best)
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
			

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			# if args.save_model: policy.save(f"./models/{file_name}")
			policy.save(f"./models/{file_name}")   
	policy.save(f"./models/{file_name}")

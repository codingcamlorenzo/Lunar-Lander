import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load("lunar_lander_penalty")   # change this name per model

eval_env = Monitor(gym.make("LunarLander-v3"))

rewards, lengths = evaluate_policy(
    model, eval_env,
    n_eval_episodes=20,
    deterministic=True,
    return_episode_rewards=True,
)

print(f"mean_reward = {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
print(f"mean_length = {np.mean(lengths):.1f} steps")
print(f"landings (reward >= 200): {sum(r >= 200 for r in rewards)}/{len(rewards)}")

eval_env.close()
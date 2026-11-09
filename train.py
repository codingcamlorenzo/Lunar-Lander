import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")             # no render_mode while training
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("lunar_lander")
env.close()



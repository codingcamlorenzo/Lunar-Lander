import gymnasium as gym
from stable_baselines3 import PPO

class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - self.penalty #keeps inbuilt reward and deducts penalty for every step in the air
        return obs, reward, terminated, truncated, info

env = TimePenaltyWrapper(gym.make("LunarLander-v3"), penalty = 0.5)             # no render_mode while training
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=750_000)
model.save("lunar_lander_penalty_0.5")
env.close()



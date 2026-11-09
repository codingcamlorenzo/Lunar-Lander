import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos_baseline",
    episode_trigger=lambda ep: True,   # record every episode
    name_prefix="baseline",
)

model = PPO.load("lunar_lander")

for episode in range(5):
    observation, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()   # important: finishes writing the last video
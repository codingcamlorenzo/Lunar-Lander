import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human") #visualising environment
observation, info = env.reset() #restarting the environment
print(f"Starting observation: {observation}")

names = ["x", "y", "vx", "vy", "angle", "ang_vel", "left_leg", "right_leg"]

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if i % 20 == 0:
        print({n: round(float(v), 2) for n, v in zip(names, observation)})

    if terminated or truncated:
        observation, info = env.reset()
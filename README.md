# Lunar-Lander: Reinforcement learning algorithms for training an agent to play the game lunar lander
A Personal Project aimed to simulate a rocket landing safely on the moon. Its done by training a PPO agent with a reward function and adds a per-frame time penalty via a custom wrapper. 

## Files and Usage 
**_run.py_:** This file creates and visualizes the LunarLander environment, where an untrained agent performs random actions at each step. The agent receives observations and rewards from the environment but doesn't know how to control the lander yet. This file serves as an example of an *untrained agent*.

**_train.py_**: This file trains the baseline agent. It creates the LunarLander environment with its built-in reward and hands it to PPO with the standard MlpPolicy network. The agent learns for 1,000,000 timesteps and the trained model is saved as lunar_lander.zip. This file produces the baseline model that every other experiment is compared against.

**_watch.py_**: This file loads the saved baseline model and records it flying. The environment is wrapped in Gymnasium’s RecordVideo wrapper, and the trained agent plays 5 episodes using its best (deterministic) action at each step. Each episode is saved as a video in the videos_baseline folder. This file serves as a visual record of the trained baseline agent.

**_evaluate.py_**: This file measures how well the baseline model performs. It loads lunar_lander and runs it for 50 episodes on the plain, unmodified environment. It prints the mean reward, mean episode length, and the number of successful landings (reward of 200 or more). This file gives the numerical score for the baseline, and the unmodified environment means every model is scored on the same original reward.

**_train_penalty.py_**: This file trains an agent on a modified reward. It defines TimePenaltyWrapper, a custom wrapper that keeps the built-in reward but subtracts a fixed penalty on every frame, so slower landings score lower. The agent is trained with a penalty of 0.5 for 750,000 timesteps and saved as lunar_lander_penalty_0.5.zip. This file is the reward-shaping experiment, testing how a per-frame time cost changes the agent’s behaviour.

**_watch_penalty.py_**: This file loads the penalty-trained model and shows it flying. It opens a live game window and runs lunar_lander_penalty_0.5 for 2000 steps, restarting after each episode. It uses the plain environment without the wrapper, since the wrapper only affects training. This file lets you visually inspect the penalty agent’s behaviour, for example to see whether it lands faster or has learned something unintended.

**_evaluate_penalty.py_**: This file measures how well the penalty-trained model performs. It runs the model for 20 episodes on the plain environment and prints the mean reward, mean episode length and landing count, using the same metrics as evaluate.py. This file gives the numerical score for the penalty agent, which you compare against the baseline.

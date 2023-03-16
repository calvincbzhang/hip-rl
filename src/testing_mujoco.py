import mujoco
import gym

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# load reacher environemnt and do 100 random steps
env = gym.make("Reacher-v4", render_mode="human")
env.reset()
for _ in range(200):
    env.step(env.action_space.sample())
    env.render()

env.close()
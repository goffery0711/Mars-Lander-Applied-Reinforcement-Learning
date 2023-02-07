import gym
import __init__         # noqa: F401
import game             # noqa: F401
from learning.agent_d import Sarsa_discreet, Sarsa_lambda_discreet
import logging


AGENT = 's_d'
episodes = 10000
eps = 1
render = False

if __name__ == '__main__':
    env = gym.make('mars_lander-v0', level=3)

    if AGENT == 's_d':
        agent = Sarsa_discreet(env, logging.DEBUG)
        agent.learn(episodes=episodes, epsilon=0.2, epsilon_decay=0.99, epsilon_min=0.01,
                    gamma=0.8, alpha=0.1, render=render)

    elif AGENT == 's_l':
        agent = Sarsa_lambda_discreet(env, logging.DEBUG)
        agent.learn(episodes=episodes, epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.05,
                    gamma=0.5, alpha=0.1, lambd=0.5, render=render)
    env.close()

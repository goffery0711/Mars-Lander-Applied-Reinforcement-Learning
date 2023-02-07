import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_reward(reward):
    '''
    Function for plotting the reward over time we smothed the rewards!
    '''
    smooth = 10
    x = range(len(reward) - (smooth-1))
    rewards_smoothed = pd.Series(reward).rolling(smooth, min_periods=smooth).mean()
    rewards_smoothed = rewards_smoothed.dropna()
    rewards_smoothed_std = pd.Series(reward).rolling(smooth, min_periods=smooth).std()
    rewards_smoothed_std = rewards_smoothed_std.dropna()
    plt.fill_between(x, rewards_smoothed - rewards_smoothed_std, rewards_smoothed + rewards_smoothed_std)
    plt.plot(x, rewards_smoothed, color="#335D7D")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward over Time (Smoothed)")
    plt.show()

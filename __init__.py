from gym.envs.registration import register

register(
    id='mars_lander-v0',
    entry_point='game.mars_lander:MarsLander',
)

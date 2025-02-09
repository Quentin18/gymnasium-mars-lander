from gymnasium.envs.registration import register

register(
    id="gymnasium_mars_lander/MarsLander-v0",
    entry_point="gymnasium_mars_lander.envs:MarsLanderEnv",
    max_episode_steps=2000,
)

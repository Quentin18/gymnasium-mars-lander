from gymnasium.envs.registration import register

register(
    id="gymnasium_mars_lander/MarsLander-v1",
    entry_point="gymnasium_mars_lander.envs:MarsLanderEnv",
    max_episode_steps=2000,
)

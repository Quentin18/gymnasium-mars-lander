# gymnasium-mars-lander

Gymnasium environment for the Mars Lander CG puzzle

## Train an agent

```bash
python -m rl_zoo3.train \
  --algo ppo \
  --env gymnasium_mars_lander/MarsLander-v0 \
  --log-interval 100 \
  --eval-freq 10000 \
  --eval-episodes 100 \
  --seed 42 \
  --gym-packages gymnasium_mars_lander \
  --conf-file hyperparams/ppo.yml \
  --progress \
  --track \
  --wandb-project-name mars-lander
```

## Enjoy a trained agent

```bash
python -m rl_zoo3.enjoy \
  --algo ppo \
  --env gymnasium_mars_lander/MarsLander-v0 \
  --n-timesteps 5000 \
  --deterministic \
  --seed 42 \
  --gym-packages gymnasium_mars_lander \
  --progress
```

## References

- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Mars Lander with Reinforcement Learning](https://github.com/antoinebrl/rl-mars-lander)

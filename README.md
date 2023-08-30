# gymnasium-mars-lander

Gymnasium environment for the Mars Lander CG puzzle

## Train

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

## References

- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Mars Lander with Reinforcement Learning](https://github.com/antoinebrl/rl-mars-lander)

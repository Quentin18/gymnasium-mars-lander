#!/usr/bin/env bash

EPISODE="${1:-2}"
ENV_NAME="${2:-MarsLander-v1}"
LOG_DIR="logs"

echo "Train episode ${EPISODE}"

case "$EPISODE" in
  1)
    STEPS=(
      "-1 3000000"
    )
    ;;
  2)
    STEPS=(
      "0 5000000 "
      "1 5000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_1/best_model.zip"
      "2 5000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_2/best_model.zip"
      "3 5000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_3/best_model.zip"
      "-1 20000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_4/best_model.zip"
    )
    ;;
  3)
    STEPS=(
      "0 2000000 "
      "1 2000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_1/best_model.zip"
      "2 5000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_2/best_model.zip"
      "3 5000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_3/best_model.zip"
      "-1 10000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_4/best_model.zip"
      "-1 2000000 ${LOG_DIR}/ppo/gymnasium_mars_lander-${ENV_NAME}_5/best_model.zip"
    )
    ;;
esac

for step in "${STEPS[@]}"; do
  read -r start timesteps model <<< "$step"

  extra_args=()
  if [[ -n "$model" ]]; then
    extra_args+=("-i" "$model")
  fi

  echo "=== Start training: start=${start}, timesteps=${timesteps} ==="

  python -m rl_zoo3.train \
    --algo ppo \
    --env "gymnasium_mars_lander/${ENV_NAME}" \
    --tensorboard-log "$LOG_DIR" \
    --n-timesteps "$timesteps" \
    --log-interval 100 \
    --eval-freq 100000 \
    --eval-episodes 20 \
    --seed 42 \
    --gym-packages gymnasium_mars_lander \
    --conf-file hyperparams/ppo.yml \
    --progress \
    --env-kwargs "episode:int(${EPISODE})" "start:int(${start})" "sequential_maps:True" \
    "${extra_args[@]}"

done

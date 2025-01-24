#!/bin/bash

# ./sh/batch_eval.sh -e false -s "best-checkpoint-0" -b "train/train_5/chall_mt_train" -g "eval_train_5"
# ./sh/batch_eval.sh -e false -s "checkpoint-1800" -b "train/train_4/chall_mt_train" -g "eval_split2_1800"

# ./sh/batch_eval.sh -e true -s "best-checkpoint-0" -b "train/train_/chall_mt_train" -g "eval_split2"
# ./sh/batch_eval.sh -e true -s "checkpoint-1800" -b "train/train_4/chall_mt_train" -g "eval_split2_1800"


# Default values for CLI arguments
EXECUTE_MODE=false
CHECKPOINT_SUFFIX="best-checkpoint-0"
CHECKPOINT_BASE="train/train_4/chall_mt_train"
ALT_BASE_PATH="../chall_mt/models/"
GROUP="eval"  # Default group

# Base configuration
CONFIG_REAL="config/eval/eval-real-config-split2-defaults.yaml"
CONFIG_SYNTH="config/eval/eval-synth-config-defaults.yaml"
PREFIX_REAL="real"
PREFIX_SYNTH="synth"

# Define ranges
REAL_DATA_VALUES=("0" "20" "40" "60")
SYNTH_DATA_VALUES=("0" "20" "40" "60" "80" "100")

# Parse CLI arguments
while getopts "e:s:b:g:" opt; do
  case $opt in
    e)
      EXECUTE_MODE=$OPTARG
      ;;
    s)
      CHECKPOINT_SUFFIX=$OPTARG
      ;;
    b)
      CHECKPOINT_BASE=$OPTARG
      ;;
    g)
      GROUP=$OPTARG
      ;;
    *)
      echo "Usage: $0 [-e <true|false>] [-s <checkpoint_suffix>] [-b <checkpoint_base>] [-g <group_name>]"
      exit 1
      ;;
  esac
done

# Function to run evaluations
run_evaluations() {
  local config=$1
  local prefix=$2
  local real=$3

  for synth in "${SYNTH_DATA_VALUES[@]}"; do
    experiment_tag="${prefix}_${real}_${synth}"
    checkpoint="${CHECKPOINT_BASE}_${real}_${synth}__5/${CHECKPOINT_SUFFIX}"
    alt_checkpoint="${ALT_BASE_PATH}${checkpoint}"

    if [[ -d "$alt_checkpoint" ]]; then
      if [[ "$EXECUTE_MODE" == true ]]; then
        python eval.py --config "$config" --group "$GROUP" --experiment_tag "$experiment_tag" --checkpoint "$checkpoint"
      else
        echo "python eval.py --config \"$config\" --group \"$GROUP\" --experiment_tag \"$experiment_tag\" --checkpoint \"$checkpoint\""
      fi
    else
      echo "Skipping: Checkpoint not found for experiment_tag \"$alt_checkpoint\""
    fi
  done
}

# Run Real Evaluations
for real in "${REAL_DATA_VALUES[@]}"; do
  run_evaluations "$CONFIG_REAL" "$PREFIX_REAL" "$real"
done

# Run Synth Evaluations
for real in "${REAL_DATA_VALUES[@]}"; do
  run_evaluations "$CONFIG_SYNTH" "$PREFIX_SYNTH" "$real"
done

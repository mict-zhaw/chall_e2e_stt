#!/bin/bash


# 0_XX
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_0_20 --checkpoint train/train_3/chall_mt_train_0_20/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_0_40 --checkpoint train/train_3/chall_mt_train_0_40/checkpoint-2700
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_0_60 --checkpoint train/train_3/chall_mt_train_0_60/checkpoint-2300
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_0_80 --checkpoint train/train_3/chall_mt_train_0_80/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_0_100 --checkpoint train/train_3/chall_mt_train_0_100/checkpoint-1800

# 20_XX
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_20_0 --checkpoint train/train_3/chall_mt_train_20_0/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_20_20 --checkpoint train/train_3/chall_mt_train_20_20/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_20_40 --checkpoint train/train_3/chall_mt_train_20_40/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_20_60 --checkpoint train/train_3/chall_mt_train_20_60/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_20_80 --checkpoint train/train_3/chall_mt_train_20_80/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_20_100 --checkpoint train/train_3/chall_mt_train_20_100/checkpoint-1800

# 40_XX
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_40_0 --checkpoint train/train_3/chall_mt_train_40_0/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_40_20 --checkpoint train/train_3/chall_mt_train_40_20/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_40_40 --checkpoint train/train_3/chall_mt_train_40_40/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_40_60 --checkpoint train/train_3/chall_mt_train_40_60/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_40_80 --checkpoint train/train_3/chall_mt_train_40_80/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_40_100 --checkpoint train/train_3/chall_mt_train_40_100/checkpoint-1800

# 60_XX
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_60_0 --checkpoint train/train_3/chall_mt_train_60_0/checkpoint-1800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_60_20 --checkpoint train/train_3/chall_mt_train_60_20/checkpoint-3000
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_60_40 --checkpoint train/train_3/chall_mt_train_60_40/checkpoint-3000
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_60_60 --checkpoint train/train_3/chall_mt_train_60_60/checkpoint-2900
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_60_80 --checkpoint train/train_3/chall_mt_train_60_80/checkpoint-2800
python eval.py --config config/eval/eval-synth-config-defaults.yaml --group eval_synth --experiment_tag eval_synth_60_100 --checkpoint train/train_3/chall_mt_train_60_100/checkpoint-1800

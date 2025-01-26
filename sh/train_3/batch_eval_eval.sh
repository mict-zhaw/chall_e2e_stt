#!/bin/bash


# XX_0
python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_20_0 --checkpoint train/train_3/chall_mt_train_20_0/best-checkpoint-0
python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_40_0 --checkpoint train/train_3/chall_mt_train_40_0/best-checkpoint-0
python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_60_0 --checkpoint train/train_3/chall_mt_train_60_0/best-checkpoint-0

# XX_20
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_0_20 --checkpoint train/train_3/chall_mt_train_0_20/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_20_20 --checkpoint train/train_3/chall_mt_train_20_20/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_40_20 --checkpoint train/train_3/chall_mt_train_40_20/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_60_20 --checkpoint train/train_3/chall_mt_train_60_20/best-checkpoint-0

# XX_40
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_0_40 --checkpoint train/train_3/chall_mt_train_0_40/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_20_40 --checkpoint train/train_3/chall_mt_train_20_40/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_40_40 --checkpoint train/train_3/chall_mt_train_40_40/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_60_40 --checkpoint train/train_3/chall_mt_train_60_40/best-checkpoint-0

# XX_60
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_0_60 --checkpoint train/train_3/chall_mt_train_0_60/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_20_60 --checkpoint train/train_3/chall_mt_train_20_60/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_40_60 --checkpoint train/train_3/chall_mt_train_40_60/best-checkpoint-0
python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_60_60 --checkpoint train/train_3/chall_mt_train_60_60/best-checkpoint-0

# XX_80
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_0_80 --checkpoint train/train_3/chall_mt_train_0_80/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_20_80 --checkpoint train/train_3/chall_mt_train_20_80/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_40_80 --checkpoint train/train_3/chall_mt_train_40_80/best-checkpoint-0
python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_60_80 --checkpoint train/train_3/chall_mt_train_60_80/best-checkpoint-0

# XX_100
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_0_100 --checkpoint train/train_3/chall_mt_train_0_100/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_20_100 --checkpoint train/train_3/chall_mt_train_20_100/best-checkpoint-0
#python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_40_100 --checkpoint train/train_3/chall_mt_train_40_100/best-checkpoint-0
python eval.py --config config/eval/eval-eval-config-defaults.yaml --group eval_eval --experiment_tag real_60_100 --checkpoint train/train_3/chall_mt_train_60_100/best-checkpoint-0

import json
import os

from src.models.config.base_config import BaseConfig

# Defaults
_DEFAULT_TRAIN_CORPORA = [["chall_train", "train"]]
_DEFAULT_EVAL_CORPORA = [["chall_train", "train"]]
_DEFAULT_EXPERIMENT_NAME = "ChaLL"
_DEFAULT_EXPERIMENT_TAG = "getting_started"
_DEFAULT_SEED = 123
_DEFAULT_N_STEPS = 30
_DEFAULT_DEVICE = "cuda"
_DEFAULT_BATCH_SIZE = 2
_DEFAULT_EVAL_BATCH_SIZE = 2
_DEFAULT_GRADIENT_ACCUMULATION_STEPS = 90
_DEFAULT_GRADIENT_CHECKPOINTING = 1
_DEFAULT_LEARNING_RATE = 3e-05
_DEFAULT_VALIDATION_FREQ = 100
_DEFAULT_SAVE_STEP = 100
_DEFAULT_LOGGING_STEP = 10
_DEFAULT_CHECKPOINT = None
_DEFAULT_N_VALID_SAMPLES = -1
_DEFAULT_N_TRAIN_SAMPLES = -1
_DEFAULT_ALT_BASE_PATH = "./"
_DEFAULT_LABEL_SMOOTHING_FACTOR = 0.0
_DEFAULT_WAV2VEC_BASE_MODEL = "facebook/wav2vec2-xls-r-300m"
_DEFAULT_BF16 = False
_DEFAULT_FP16 = True
_DEFAULT_OPTIM = "adamw_hf"
_DEFAULT_FREEZE_W2VM = False
_DEFAULT_SAVE_TOTAL_LIMIT = 2


class TrainConfig(BaseConfig):
    def __init__(self, **kwargs):
        self.train_corpora = kwargs.pop('train_corpora', _DEFAULT_TRAIN_CORPORA)
        self.eval_corpora = kwargs.pop('eval_corpora', _DEFAULT_EVAL_CORPORA)
        self.experiment_name = kwargs.pop('experiment_name', _DEFAULT_EXPERIMENT_NAME)
        self.experiment_tag = kwargs.pop('experiment_tag', _DEFAULT_EXPERIMENT_TAG)
        self.n_steps = kwargs.pop('n_steps', _DEFAULT_N_STEPS)
        self.device = kwargs.pop('device', _DEFAULT_DEVICE)
        self.batch_size = kwargs.pop('batch_size', _DEFAULT_BATCH_SIZE)
        self.eval_batch_size = kwargs.pop('eval_batch_size', _DEFAULT_EVAL_BATCH_SIZE)
        self.gradient_accumulation_steps = kwargs.pop('gradient_accumulation_steps', _DEFAULT_GRADIENT_ACCUMULATION_STEPS)
        self.gradient_checkpointing = kwargs.pop('gradient_checkpointing', _DEFAULT_GRADIENT_CHECKPOINTING)
        self.learning_rate = kwargs.pop('learning_rate', _DEFAULT_LEARNING_RATE)
        self.validation_freq = kwargs.pop('validation_freq', _DEFAULT_VALIDATION_FREQ)
        self.save_step = kwargs.pop('save_step', _DEFAULT_SAVE_STEP)
        self.logging_step = kwargs.pop('logging_step', _DEFAULT_LOGGING_STEP)
        self.checkpoint = kwargs.pop('checkpoint', _DEFAULT_CHECKPOINT)
        self.n_valid_samples = kwargs.pop('n_valid_samples', _DEFAULT_N_VALID_SAMPLES)
        self.n_train_samples = kwargs.pop('n_train_samples', _DEFAULT_N_TRAIN_SAMPLES)
        self.alt_base_path = kwargs.pop('alt_base_path', _DEFAULT_ALT_BASE_PATH)
        self.label_smoothing_factor = kwargs.pop('label_smoothing_factor', _DEFAULT_LABEL_SMOOTHING_FACTOR)
        self.wav2vec_base_model = kwargs.pop('wav2vec_base_model', _DEFAULT_WAV2VEC_BASE_MODEL)
        self.bf16 = kwargs.pop('bf16', _DEFAULT_BF16)
        self.fp16 = kwargs.pop('fp16', _DEFAULT_FP16)
        self.optim = kwargs.pop('optim', _DEFAULT_OPTIM)
        self.freeze_w2vm = kwargs.pop('freeze_w2vm', _DEFAULT_FREEZE_W2VM)
        self.save_total_limit = kwargs.pop('save_total_limit', _DEFAULT_SAVE_TOTAL_LIMIT)
        self.seed = kwargs.pop("seed", _DEFAULT_SEED)

        self.dataset_kwargs = kwargs.pop("dataset_kwargs", {})
        self.data_dir = kwargs.pop("data_dir", "/")

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'rt', encoding='utf-8') as ifile:
            config = json.load(ifile)
        return cls(**config)

    def to_dict(self):
        return {
            'train_corpora': self.train_corpora,
            'eval_corpora': self.eval_corpora,
            'experiment_name': self.experiment_name,
            'experiment_tag': self.experiment_tag,
            'n_steps': self.n_steps,
            'device': self.device,
            'batch_size': self.batch_size,
            'eval_batch_size': self.eval_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_checkpointing': self.gradient_checkpointing,
            'learning_rate': self.learning_rate,
            'validation_freq': self.validation_freq,
            'save_step': self.save_step,
            'logging_step': self.logging_step,
            'checkpoint': self.checkpoint,
            'n_valid_samples': self.n_valid_samples,
            'n_train_samples': self.n_train_samples,
            'alt_base_path': self.alt_base_path,
            'label_smoothing_factor': self.label_smoothing_factor,
            'wav2vec_base_model': self.wav2vec_base_model,
            'bf16': self.bf16,
            'fp16': self.fp16,
            'optim': self.optim,
            'freeze_w2vm': self.freeze_w2vm,
            'save_total_limit': self.save_total_limit,
            'seed': self.seed,
            'dataset_kwargs': self.dataset_kwargs,
            'data_dir': self.data_dir
        }

    def to_json(self, file_path: str, file_name: str = 'config.json'):
        with open(os.path.join(file_path, file_name), 'wt', encoding='utf-8') as out_file:
            json.dump(self.to_dict(), out_file)

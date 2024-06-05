from typing import List

import numpy as np
import random
import torch
import wandb
import json
import os
from collections import Counter

from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
import evaluate

from src.data_collator.data_collator import DataCollatorCTCWithPadding
from src.models.config.train_config import TrainConfig

from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.models.wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
from transformers import TrainingArguments, SchedulerType, PreTrainedModel
from transformers.trainer import Trainer

from src.train_logger import TrainLogger
from src.utils.bleu_metric import score

wer_metric = evaluate.load("wer")
bleu_metric = evaluate.load('sacrebleu')

_DEFAULT_ENV = "production"


class Wav2VecPipeline:
    env: str
    logger: TrainLogger

    config: TrainConfig
    dataset: DatasetDict

    logging_path: str
    base_model_path: str

    vocab_file: str
    processor: Wav2Vec2Processor
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer
    model: PreTrainedModel
    training_args: TrainingArguments
    data_collator: DataCollatorCTCWithPadding

    save_nr: int = 1
    best_wer: float = 1
    label_feature: str = "clear_text"

    accelerate: Accelerator = Accelerator()

    def __init__(self, config_file: str = None, env: str = "production"):
        """
        Initialize the Wav2VecPipeline with the given configuration file.

        :param config_file: Path to the configuration file.
        :param env: Environment name
        """

        self.config = TrainConfig.from_json(config_file) if config_file else TrainConfig()
        self.env = env
        random.seed(self.config.seed)

        # Define data dir
        self.data_dir = str(os.path.join(self.config.alt_base_path, self.config.data_dir))
        self.cache_dir = str(os.path.join(self.config.alt_base_path, "cache"))
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Setup model storage
        self.base_model_path = str(os.path.join(self.config.alt_base_path, 'models', self.config.experiment_name, self.config.experiment_tag))
        self.vocab_file = os.path.join(self.base_model_path, 'vocab.json')
        if not os.path.exists(self.base_model_path):
            os.makedirs(self.base_model_path, exist_ok=True)

        # Setup logging
        self.logging_path = str(os.path.join(self.config.alt_base_path, 'logging', self.config.experiment_name, self.config.experiment_tag))
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path, exist_ok=True)
        self.logger = TrainLogger(self.logging_path)
        self.config.to_json(self.logging_path)

        # Prepared Data Path
        self.prepared_data_path = os.path.join(self.cache_dir, "tokenized_data", self.config.experiment_name, self.config.experiment_tag)
        if not os.path.exists(self.prepared_data_path):
            os.makedirs(self.prepared_data_path, exist_ok=True)

    def save_dataset(self, path: str):
        """
        Save the dataset to disk.

        :param path: Directory path where the dataset will be saved.
        """
        self.logger.log_event("Save Dataset", path=path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.dataset.save_to_disk(path)
        self.logger.log_event("Dataset Saved")

    def load_dataset(self, path: str):
        """
        Load the dataset from disk.

        :param path: Directory path from where the dataset will be loaded.
        """
        self.logger.log_event("Load Dataset", path=path)
        self.dataset = DatasetDict.load_from_disk(path)
        self.logger.log_event("Dataset Loaded")

    def run(self) -> None:
        """
        Run the training pipeline.
        """

        self.logger.log_event("Run", env=self.env)

        self.load_data(self.config.train_corpora, self.config.eval_corpora, self.config.dataset_kwargs)
        self.create_vocabulary_file()

        self.tokenizer = self.create_tokenizer()
        self.processor = self.create_processor()

        self.prepare_dataset()

        self.train()

    def run_kfold(self):

        _all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _folds = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]

        self.create_vocabulary_file()

        self.create_tokenizer()
        self.create_processor()

        for fold in _folds:
            train_corpora = [["corpora", list(set(_all) - set(fold))]]
            eval_corpora = [["corpora", list(set(fold))]]
            self.load_data(train_corpora, eval_corpora, self.config.dataset_kwargs)
            self.prepare_dataset()

            self.train()
            # do the rest as before

    def load_data(self, train_corpora: List, eval_corpora: List, data_kwargs: dict = None):
        """
        Loads and processes training and evaluation datasets.
        This method takes lists of training and evaluation corpora, and concatenates them into single training and evaluation datasets.
        If training and evaluation corpora contains multiple splits they are also concatenated into a single training and evaluation dataset.

        @param train_corpora: A list of tuples where each tuple contains the name of the training corpus and the split.
                              Example: [('dataset_name1', 'train'), ('dataset_name2', 'train')]
        @param eval_corpora: A list of tuples where each tuple contains the name of the evaluation corpus and the split.
                             Example: [('dataset_name1', 'validation'), ('dataset_name2', 'test')]
        @param data_kwargs: A dictionary of additional arguments to pass to the `load_dataset` function.
        """

        self.logger.log_event("Load data", train_corpora=train_corpora, eval_corpora=eval_corpora, **data_kwargs)
        assert os.path.exists(self.data_dir), "Data directory does not exist"

        cache_dir = os.path.join(self.cache_dir, "dataset")

        with self.accelerate.main_process_first():

            # todo this breaks hf map caching, but its not relevant for now anyway
            # train_datasets = []
            # for corpus, split in train_corpora:
            #     train_ds = load_dataset(corpus, split=split, data_dir=self.data_dir, cache_dir=cache_dir, trust_remote_code=True, **data_kwargs)
            #     if type(train_ds) is DatasetDict:
            #         train_datasets.append(concatenate_datasets(list(train_ds)))
            #     else:
            #         train_datasets.append(train_ds)
            #
            # eval_datasets = []
            # for corpus, split in eval_corpora:
            #     eval_ds = load_dataset(corpus, split=split, data_dir=self.data_dir, cache_dir=cache_dir, trust_remote_code=True, **data_kwargs)
            #     if type(eval_ds) is DatasetDict:
            #         eval_datasets.append(concatenate_datasets(list(eval_ds)))
            #     else:
            #         eval_datasets.append(eval_ds)

            test = load_dataset(
                    train_corpora[0][0],
                    # split=train_corpora[0][1],
                    data_dir=self.data_dir,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    folds={"train": ["2"]},
                    stratify_column="intervention"
                )

            print(test["train"][0])
            print(test["train"][7])
            print(test["train"][20])


            self.dataset = DatasetDict({
                "train": load_dataset(
                    train_corpora[0][0],
                    split=train_corpora[0][1],
                    data_dir=self.data_dir,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    **data_kwargs
                ),
                "eval": load_dataset(
                    eval_corpora[0][0],
                    split=eval_corpora[0][1],
                    data_dir=self.data_dir,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    **data_kwargs
                ),
            })

            print(self.dataset["train"][0])

    def create_vocabulary_file(self, min_freq: int = 1):
        """
        Create a vocabulary file from the dataset.

        :param min_freq: Minimum frequency of characters to be included in the vocabulary.
        """
        self.logger.log_event("Create Vocabulary File", min_freq=min_freq)
        assert self.vocab_file, f"Vocabulary file path not defined"

        # Check if the vocabulary file already exists
        if os.path.exists(self.vocab_file):
            self.logger.log_event("Vocabulary File Exists", path=self.vocab_file)
            return

        os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
        vocab = self._create_vocabulary(min_freq=min_freq)

        with open(self.vocab_file, 'wt', encoding='utf-8') as voc_file:
            json.dump(vocab, voc_file)

        self.logger.log_event("Vocabulary File Created", total_chars=len(vocab), vocab=vocab)

    def _create_vocabulary(self, min_freq: int = 10, split: str = "train", feature: str = "clear_text") -> dict:
        """
        Create a character-level vocabulary from the dataset.

        :param min_freq: Minimum frequency of characters to be included in the vocabulary.
        :param split: Dataset split to use for creating the vocabulary.
        :param feature: Feature name in the dataset to be used for creating the vocabulary.
        :return: Dictionary of characters and their indices.
        """
        vocab = Counter()
        for clear_text in self.dataset[split][feature]:
            vocab.update(clear_text)

        char2idx = {c: idx for idx, (c, freq) in enumerate(vocab.items(), start=1) if freq > min_freq}

        if " " in char2idx:
            char2idx["|"] = char2idx.pop(" ")
        char2idx["[UNK]"] = len(char2idx)

        return char2idx

    def create_tokenizer(self, unk_token: str = "[UNK]", pad_token: str = "[PAD]", word_delimiter_token: str = "|") -> Wav2Vec2CTCTokenizer:
        """
        Create a tokenizer for the Wav2Vec2 model.

        :param unk_token: Token for unknown words.
        :param pad_token: Token for padding.
        :param word_delimiter_token: Token for word delimiter.
        """
        self.logger.log_event("Create Tokenizer", unk_token=unk_token, pad_token=pad_token, word_delimiter_token=word_delimiter_token)
        assert os.path.exists(self.vocab_file), f"Vocabulary file not found at {self.vocab_file}"

        tokenizer = Wav2Vec2CTCTokenizer(
            self.vocab_file,
            unk_token=unk_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter_token
        )
        tokenizer.add_tokens([pad_token, unk_token])
        return tokenizer

    def create_processor(self, feature_size: int = 1, sampling_rate: int = 16_000) -> Wav2Vec2Processor:
        """
        Create a feature extractor for the Wav2Vec2 model.

        @param feature_size: Size of the feature to be extracted.
        @param sampling_rate: Sampling rate of the audio.
        """
        self.logger.log_event("Create Feature Extractor", feature_size=feature_size, sampling_rate=sampling_rate)
        assert self.tokenizer, "Tokenizer not initialized"

        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=self.create_tokenizer())

    def prepare_dataset(self) -> None:
        """
        Prepare the dataset for training and evaluation.
        """

        self.logger.log_event("Prepare Data")

        processor = self.create_processor()
        label = self.label_feature
        remove_columns = self.dataset.column_names["train"]

        def _prepare_dataset(batch):

            audio = batch["audio"]
            batch["input_length"] = len(audio["array"])

            # batched output is "un-batched" to ensure mapping is correct
            batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

            with processor.as_target_processor():
                batch["labels"] = processor(batch[label]).input_ids

            return batch

        # Do data preparation on first process and load from cache in other
        with self.accelerate.main_process_first():

            self.dataset = self.dataset.map(_prepare_dataset, remove_columns=remove_columns, num_proc=4, load_from_cache_file=True)

            if self.config.n_train_samples > 0:
                self.dataset["train"] = self.dataset["train"].shuffle(seed=self.config.seed).select(range(self.config.n_train_samples))
            else:
                self.dataset["train"] = self.dataset["train"].shuffle(seed=self.config.seed)

            if self.config.n_valid_samples > 0:
                self.dataset["eval"] = self.dataset["eval"].shuffle(seed=self.config.seed).select(range(self.config.n_valid_samples))
            else:
                self.dataset["eval"] = self.dataset["eval"].shuffle(seed=self.config.seed)

        self.logger.log_event("Data Prepared", num_train=len(self.dataset["train"]), num_valid=len(self.dataset["eval"]))

    def _get_model(self) -> Wav2Vec2ForCTC:
        """
        Load or initialize the Wav2Vec2 model.

        :return: Wav2Vec2ForCTC model.
        """
        model_path = None
        if self.config.checkpoint is not None:
            model_path = os.path.join(self.base_model_path)
            model_name = os.path.join(model_path, self.config.checkpoint)
        else:
            model_name = self.config.wav2vec_base_model

        assert self.processor, "No processor"

        self.logger.log_event("Load Model", model_name=model_name, model_path=model_path)

        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            cache_dir=os.path.join(self.cache_dir, "models"),
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.tokenizer),
        )

        model.config.ctc_zero_infinity = True
        model.freeze_feature_extractor()
        if self.config.freeze_w2vm:
            for module in model.wav2vec2.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        perc = 100 * (trainable_params / total_params)

        self.logger.log_event("Model Loaded", model_name=model_name,
                              pytorch_trainable_params=trainable_params, pytorch_total_params=total_params, perc=perc)
        return model

    def _get_train_arguments(self) -> TrainingArguments:
        """
        Get training arguments for the Wav2Vec2 model.

        :return: TrainingArguments.
        """

        no_cuda = not self.config.device == 'cuda'

        return TrainingArguments(
            output_dir=self.base_model_path,
            logging_dir=self.logging_path,
            group_by_length=True,
            length_column_name="input_length",
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            evaluation_strategy="steps",
            num_train_epochs=self.config.n_steps,
            no_cuda=no_cuda,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            bf16_full_eval=self.config.bf16,
            fp16_full_eval=self.config.fp16,
            save_steps=self.config.save_step,
            eval_steps=self.config.validation_freq,
            logging_steps=self.config.logging_step,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            optim=self.config.optim,
            ddp_find_unused_parameters=True,
            lr_scheduler_type=SchedulerType.LINEAR,
            ignore_data_skip=True,
            label_smoothing_factor=self.config.label_smoothing_factor,
            seed=self.config.seed,
            save_total_limit=self.config.save_total_limit
        )

    def train(self):
        """
        Train the Wav2Vec2 model.
        """
        assert self.processor, "Processor not initialized"
        assert self.tokenizer, "Tokenizer not initialized"

        self.model = self._get_model()
        self.training_args = self._get_train_arguments()
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.logger.log_event("Start Train", is_cuda_available=torch.cuda.is_available(),
                              batch_size=self.config.batch_size, gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                              learning_rate=self.config.learning_rate, bf16=self.config.bf16, fp16=self.config.fp16, optim=self.config.optim)

        wandb.init(
            project='chall_dev' if self.env == 'development' else 'chall',
            entity="chall",
            config=self.config.to_dict(),
            name=f'{self.config.experiment_name}_{self.config.experiment_tag}',
            group=self.config.experiment_tag
        )

        self.logger.log_event("Setup Trainer")

        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"],
            tokenizer=self.processor.feature_extractor,
            compute_metrics=self.compute_metrics
        )

        self.logger.log_event("Start  Trainer")

        ignore_keys_for_eval = ['past_key_values', 'encoder_last_hidden_state', 'hidden_states', 'cross_attentions']
        train_res = trainer.train(resume_from_checkpoint=False, ignore_keys_for_eval=ignore_keys_for_eval)

        self.logger.log_event("Train End", train_res=train_res)

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics for the predictions.

        @param pred: Predictions from the model.
        @return: Dictionary of evaluation metrics.
        """
        pred_logits = pred.predictions
        if pred_logits is None:
            self.logger.log_event("Empty pred_logits received")
            return {"wer": float('inf'), "bleu": 0.0}

        pred_ids = np.argmax(pred_logits, axis=2)
        if pred_ids is None:
            self.logger.log_event("Empty label_ids received")
            return {"wer": float('inf'), "bleu": 0.0}

        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id

        pred_strs = self.tokenizer.batch_decode(pred_ids, group_tokens=True)
        label_strs = self.tokenizer.batch_decode(pred.label_ids, group_tokens=False)  # we do not want to group tokens when computing the metrics

        # Check if pred_strs or label_strs are empty
        if not pred_strs:
            self.logger.log_event("Empty predictions or labels after decoding")
            return {"wer": float('inf'), "bleu": 0.0}

        wer = wer_metric.compute(predictions=pred_strs, references=label_strs)
        bleu_score = score(pred_strs, label_strs)

        self.log_metrics(wer, bleu_score, pred_ids, pred_strs, label_strs)

        return {"wer": wer, "bleu": bleu_score}

    def log_metrics(self, wer, bleu_score, pred_ids, pred_strs, label_strs, wandb_save: bool = True):
        """
        Log evaluation metrics to files.

        @param wer: Word Error Rate.
        @param bleu_score: BLEU score.
        @param pred_ids: Predicted IDs.
        @param pred_strs: Predicted strings.
        @param label_strs: Label strings.
        @param wandb_save: Save files to wandb
        """
        try:
            with open(os.path.join(self.logging_path, f'output_{self.save_nr}.txt'), 'wt', encoding='utf-8') as log_file:
                log_file.write(f'WER: {wer}\t BLEU:{0}\n')
                for idx, (pred_str, label_str) in enumerate(zip(pred_strs, label_strs)):
                    log_file.write(f'{idx}\t{pred_str}\t{label_str}\n')

            with open(os.path.join(self.logging_path, f'ctc_output_{self.save_nr}.txt'), 'wt', encoding='utf-8') as ctc_file:
                ctc_file.write(f'WER: {wer}\t BLEU:{bleu_score}\n')
                for idx, _ in enumerate(pred_ids):
                    ctc_tokens = " ".join(self.tokenizer.convert_ids_to_tokens(pred_ids[idx].tolist()))
                    ctc_file.write(f'{idx}\t{ctc_tokens}\n')

            self.save_nr += 1

            if wer < self.best_wer:
                self.best_wer = wer
                opath = os.path.join(self.base_model_path, f'best-checkpoint-0')
                self.model.save_pretrained(opath)

                self.logger.log_event("Model saved", path=opath)

            # Optionally, log the files to wandb as well
            if wandb_save:
                wandb.save(os.path.join(self.logging_path, f'output_{self.save_nr}.txt'))
                wandb.save(os.path.join(self.logging_path, f'ctc_output_{self.save_nr}.txt'))

        except Exception as e:
            self.logger.log_event("Error in log_metrics", error=str(e))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Process configuration file for dataset creation.')
    # parser.add_argument('config_file', type=str, help='Path to the configuration file', default='config/train/config_remote.json', nargs="?")
    # args = parser.parse_args()

    env = os.environ.get('ENV', _DEFAULT_ENV)
    pipeline = Wav2VecPipeline(config_file=f'config/train/config_{env}.json', env=env)
    pipeline.run()

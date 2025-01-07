import json
import os
import random
import re
from collections import Counter
from typing import List

import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, load_from_disk
from transformers import TrainingArguments, SchedulerType, PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.models.wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from transformers.trainer import Trainer

import wandb
from src.data_collator.data_collator import DataCollatorCTCWithPadding
from src.models.config.train_config import TrainConfig, CorpusConfig
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
    label_feature: str = "text_label"

    accelerator: Accelerator

    def __init__(self, config: TrainConfig, env: str = "production"):
        """
        Initialize the Wav2VecPipeline with the given configuration file.

        :param config_file: Path to the configuration file.
        :param env: Environment name
        """

        self.config = config
        self.env = env
        random.seed(self.config.seed)

        # Define data dir
        self.cache_dir = str(os.path.join(self.config.alt_base_path, ".cache"))
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Setup model storage
        self.base_model_path = str(os.path.join(self.config.alt_base_path, 'models', config.group, config.job_type, config.experiment_label))
        self.vocab_file = os.path.join(self.base_model_path, 'vocab.json')
        if not os.path.exists(self.base_model_path):
            os.makedirs(self.base_model_path, exist_ok=True)

        # Setup logging
        self.logging_path = str(os.path.join(self.config.alt_base_path, 'logs', "runs", config.group, config.job_type, config.run_id or ""))
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path, exist_ok=True)
        self.logger = TrainLogger(self.logging_path)
        self.config.to_json(self.logging_path)

        # Prepared Data Path
        # self.prepared_data_path = os.path.join(self.cache_dir, "tokenized_data", config.group, config.job_type, config.experiment_label)
        # if not os.path.exists(self.prepared_data_path):
        #     os.makedirs(self.prepared_data_path, exist_ok=True)

        self.accelerator = Accelerator()

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

    def run(self) -> None:
        """
        Run the training pipeline.
        """

        self.logger.log_event(
            "Start Run",
            env=self.env,
            alt_base_path=config.alt_base_path,
            job_type=config.job_type,
            experiment_label=config.experiment_label,
            run_id=config.run_id,
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            wandb_entity=os.environ.get("WANDB_ENTITY"),
            wandb_project=os.environ.get("WANDB_PROJECT"),
            wandb_dir=os.environ.get("WANDB_DIR"),
            wandb_cache_dir=os.environ.get("WANDB_CACHE_DIR"),
            hf_home=os.environ.get("HF_HOME"),
            alt_base_path_env=os.environ.get("ALT_BASE_PATH"),
            rel_data_dir=os.environ.get("REL_DATA_DIR"),
        )

        dataset = self.load_data(self.config.train_corpora, self.config.eval_corpora)
        self.create_vocabulary_file(dataset)

        self.tokenizer = self.create_tokenizer()
        self.processor = self.create_processor()

        dataset = self.prepare_dataset(dataset)
        self.train(dataset)

        dataset.cleanup_cache_files()

    def load_data(self, train_corpora: List[CorpusConfig], eval_corpora: List[CorpusConfig]) -> DatasetDict:
        """
        Loads datasets using Huggingface datasets.
        Splits for chall data can be defined via configs `dataset_kwargs` and `folds`.

        @param train_corpora: A list of tuples where each tuple contains the name of the training corpus and the split.
                              Example: [('dataset_name1', 'train'), ('dataset_name2', 'train')]
        @param eval_corpora: A list of tuples where each tuple contains the name of the evaluation corpus and the split.
                             Example: [('dataset_name1', 'validation'), ('dataset_name2', 'test')]
        """

        self.logger.log_event("Load data", train_corpora=[tc.model_dump() for tc in train_corpora],
                              eval_corpora=[tc.model_dump() for tc in eval_corpora])

        cache_dir = os.path.join(self.cache_dir, "dataset")
        dataset_dict = {c: DatasetDict() for c in [c.dataset for c in train_corpora + eval_corpora]}

        # do this on first and load the other from the cache
        # with self.accelerator.main_process_first():
        for corpus_config in train_corpora + eval_corpora:

            self.logger.log_event("Load Dataset", corpus=corpus_config.dataset, split=corpus_config.split)

            # Skip if target duration is zero
            if corpus_config.target_duration_hours == 0:
                continue

            # Load dataset either from disk or using load_dataset
            with self.accelerator.main_process_first():
                if corpus_config.load_from_disk:
                    ds = load_from_disk(
                        os.path.join(self.config.alt_base_path, corpus_config.dataset),
                        **corpus_config.load_dataset_kwargs
                    )
                    ds = ds[corpus_config.split] if corpus_config.split is not None else ds
                else:
                    ds = load_dataset(
                        corpus_config.dataset,
                        split=corpus_config.split,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        **corpus_config.load_dataset_kwargs
                    )

            # Normalize the columns used for training
            ds = ds.rename_columns({
                corpus_config.id_column: "audio_id",
                corpus_config.audio_column: "audio",
                corpus_config.text_column: self.label_feature
            })
            ds = ds.select_columns([col for col in ["audio_id", "audio", self.label_feature, "duration"] if col in ds.column_names])

            def calculate_duration(sample):
                try:
                    sample["duration"] = sample["audio"]['array'].shape[0] / sample["audio"]['sampling_rate']
                except Exception as e:
                    sample["duration"] = None
                    print(f"Error processing sample: {e}")
                return sample

            # Calculate duration if not existing
            if "duration" not in ds.column_names:
                with self.accelerator.main_process_first():
                    ds_durations = ds.map(calculate_duration, num_proc=1, remove_columns=["audio"], desc="Calculate Durations")
                with self.accelerator.main_process_first():
                    ds = ds.add_column("duration", ds_durations["duration"])

            # Optionally select samples to match target audio duration
            if corpus_config.target_duration_seconds is not None and corpus_config.split is not None:

                # Shuffle the data before splitting into data portions
                ds = ds.shuffle(seed=self.config.seed)

                # Find and select the indexes that result in a dataset with the target audio duration
                cumulative_duration = 0
                selected_indices = []
                for i, duration in enumerate(ds["duration"]):
                    if duration is None:
                        continue
                    if cumulative_duration + duration >= corpus_config.target_duration_seconds:
                        break
                    cumulative_duration += duration
                    selected_indices.append(i)
                ds = ds.select(selected_indices)

                self.logger.log_event("Dataset Loaded", corpus=corpus_config.dataset,
                                      split=corpus_config.split, cumulative_duration=cumulative_duration, num_samples=len(ds))

            if isinstance(ds, Dataset):
                ds = DatasetDict({corpus_config.split: ds})
            dataset_dict[corpus_config.dataset].update(ds)

        self.logger.log_event("Datasets Loaded",
                              dataset_dict=[(key, [(split, len(val[split])) for split in val]) for key, val in dataset_dict.items()])

        # Concatenate splits into one DatasetDict
        result_dataset_dict = DatasetDict({
            split_name: concatenate_datasets(
                [dataset_dict_entry[split_name]
                 for dataset_dict_entry in dataset_dict.values()
                 if split_name in dataset_dict_entry]
            )
            for split_name in {split for dataset_dict_entry in dataset_dict.values() for split in dataset_dict_entry.keys()}
        })

        # Assert that the required splits exist
        missing_splits = [split for split in ["train", "eval"] if split not in result_dataset_dict]
        assert not missing_splits, f"Missing required splits: {missing_splits}"

        # Shuffle the data to mix synth and real data
        result_dataset_dict["train"] = result_dataset_dict["train"].shuffle(seed=config.seed)
        result_dataset_dict["eval"] = result_dataset_dict["eval"].shuffle(seed=config.seed)

        # Rewrite the shuffled dataset on disk as contiguous chunks of data
        train_cache_file = os.path.join(self.cache_dir, "load_data", self.config.experiment_name, f"train_{self.config.experiment_label}.arrow")
        eval_cache_file = os.path.join(self.cache_dir, "load_data", self.config.experiment_name, f"eval_{self.config.experiment_label}.arrow")
        os.makedirs(os.path.dirname(train_cache_file), exist_ok=True)
        os.makedirs(os.path.dirname(eval_cache_file), exist_ok=True)
        result_dataset_dict["train"] = result_dataset_dict["train"].flatten_indices(cache_file_name=train_cache_file)
        result_dataset_dict["eval"] = result_dataset_dict["eval"].flatten_indices(cache_file_name=eval_cache_file)

        # Remove unused cache files when working locally. Data on scratch is deleted anyway...
        if self.env == "development":
            removed = result_dataset_dict.cleanup_cache_files()
            self.logger.log_event("Cleanup Cache Files", removed=removed)

        dataset_info = [(split, len(result_dataset_dict[split]), round(sum(result_dataset_dict[split]["duration"]) / 3600, 2)) for split in
                        result_dataset_dict]
        self.logger.log_event("Datasets Combined", dataset=dataset_info)

        return result_dataset_dict

    def create_vocabulary_file(self, dataset: DatasetDict, min_freq: int = 1):
        """
        Create a vocabulary file from the dataset.

        :param dataset: The dataset to create a vocabulary file for.
        :param min_freq: Minimum frequency of characters to be included in the vocabulary.
        """
        self.logger.log_event("Create Vocabulary File", min_freq=min_freq)
        assert self.vocab_file, f"Vocabulary file path not defined"

        # Check if the vocabulary file already exists
        if os.path.exists(self.vocab_file):
            self.logger.log_event("Vocabulary File Exists", path=self.vocab_file)
            return

        os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
        vocab = self._create_vocabulary(dataset, min_freq=min_freq)

        with open(self.vocab_file, 'wt', encoding='utf-8') as voc_file:
            json.dump(vocab, voc_file)

        self.logger.log_event("Vocabulary File Created", total_chars=len(vocab), vocab=vocab, path=self.vocab_file)

    def _create_vocabulary(self, dataset, min_freq: int = 10, feature: str = "label") -> dict:
        """
        Create a character-level vocabulary from the dataset.

        :param dataset: The dataset to create a vocabulary for.
        :param min_freq: Minimum frequency of characters to be included in the vocabulary.
        :param feature: Feature name in the dataset to be used for creating the vocabulary.
        :return: Dictionary of characters and their indices.
        """
        vocab = Counter()
        for split in dataset:
            for clear_text in dataset[split][self.label_feature]:
                vocab.update(clear_text)

        char2idx = {c: idx for idx, (c, freq) in enumerate(vocab.items(), start=1) if freq > min_freq}

        if " " in char2idx:
            char2idx["|"] = char2idx.pop(" ")
        char2idx["[UNK]"] = len(char2idx)
        char2idx["[PAD]"] = 0

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

    def prepare_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare the dataset for training and evaluation.
        """

        self.logger.log_event("Prepare Data")
        remove_columns = dataset[next(iter(dataset))].column_names

        def _prepare_dataset(batch):
            # print(f"Label: {batch.get(self.label_feature, 'MISSING')} for Audio ID: {batch.get('audio_id', 'unknown')}")

            if not batch[self.label_feature].strip():
                print("nicht gut")
                print(f"Empty label found for audio ID: {batch['audio_id']}")
                return None  # Exclude this example

            audio = batch["audio"]
            batch["input_length"] = len(audio["array"])
            # batched output is "un-batched" to ensure mapping is correct
            batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch[self.label_feature]).input_ids

            if len(batch["labels"]) == 0:
                print(f"Label: {batch['labels']}")

            # Check if labels were successfully set
            if not batch.get("labels"):
                print(f"Labels not set for audio ID: {batch.get('audio_id', 'unknown')}. Label content: {batch.get(self.label_feature, 'MISSING')}")

            return batch

        print(f"Dataset columns: {dataset.column_names}")
        print(f"Label Feature: {self.label_feature}")

        # Do data preparation on the first process and load from cache in other
        with self.accelerator.main_process_first():
            dataset = dataset.map(_prepare_dataset, remove_columns=remove_columns, num_proc=4, load_from_cache_file=True)

        self.logger.log_event("Data Prepared")
        return dataset

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
        if self.config.train_args.freeze_w2vm:
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
        train_args = self.config.train_args

        return TrainingArguments(
            output_dir=self.base_model_path,
            logging_dir=self.logging_path,
            group_by_length=True,
            length_column_name="input_length",
            per_device_train_batch_size=train_args.batch_size,
            per_device_eval_batch_size=train_args.eval_batch_size,
            gradient_accumulation_steps=train_args.gradient_accumulation_steps,
            gradient_checkpointing=train_args.gradient_checkpointing,
            evaluation_strategy="steps",
            num_train_epochs=train_args.num_train_epochs,
            max_steps=train_args.max_steps,
            no_cuda=no_cuda,
            bf16=train_args.bf16,
            fp16=train_args.fp16,
            bf16_full_eval=train_args.bf16,
            fp16_full_eval=train_args.fp16,
            save_steps=train_args.save_step,
            eval_steps=train_args.validation_freq,
            logging_steps=train_args.logging_step,
            learning_rate=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
            warmup_steps=train_args.warmup_steps,
            warmup_ratio=train_args.warmup_ratio,
            optim=train_args.optim,
            ddp_find_unused_parameters=True,
            lr_scheduler_type=SchedulerType.LINEAR,
            ignore_data_skip=True,
            label_smoothing_factor=train_args.label_smoothing_factor,
            seed=self.config.seed,
            save_total_limit=train_args.save_total_limit,
        )

    def train(self, dataset: DatasetDict):
        """
        Train the Wav2Vec2 model.
        """
        assert self.processor, "Processor not initialized"
        assert self.tokenizer, "Tokenizer not initialized"

        conf = self.config

        self.model = self._get_model()
        self.training_args = self._get_train_arguments()
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.logger.log_event("Start Train", is_cuda_available=torch.cuda.is_available(), train_args=conf.train_args.model_dump())

        settings = wandb.Settings(job_name='train_job')
        with (wandb.init(settings=settings, job_type=conf.job_type, id=config.run_id,
                         group=conf.group, config=conf.dict(), name=conf.experiment_label, resume="allow") as run):

            for corpora in self.config.train_corpora + self.config.eval_corpora:
                try:
                    artifact = re.sub(r'_v(\d+)', r':v\1', os.path.basename(corpora.dataset))
                    run.use_artifact(artifact)
                except Exception:
                    self.logger.log_event(f'Artifact not {artifact} found on WandB.', level="error")

            self.logger.log_event("Setup Trainer", steps_per_epoch=dataset["train"].shape[0] // conf.train_args.batch_size)

            trainer = Trainer(
                model=self.model,
                data_collator=self.data_collator,
                args=self.training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                tokenizer=self.processor.feature_extractor,
                compute_metrics=self.compute_metrics
            )

            self.logger.log_event("Start Trainer")

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
    # os.environ["WANDB_MODE"] = "offline"
    env = os.environ.get('ENV', _DEFAULT_ENV)

    # Combine configs to use defaults and experiment-specific configs
    # config = TrainConfig.from_yaml(args.default_config, args.config)
    config = TrainConfig.from_cli(default_config_file="config/train_chall_mt/config-defaults.yaml")
    pipeline = Wav2VecPipeline(config=config, env=env)
    pipeline.run()

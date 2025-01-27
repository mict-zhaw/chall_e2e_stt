import collections
import json
import os
import random
from typing import List, Union

from src.data_collator.data_collator import DataCollatorCTCWithPadding

collections.Iterable = collections.abc.Iterable  # used to prevent error in alignment tool
from alignment_tool.alignment.extensions.annotation_mutation_evaluation import AnnotationEvaluation

import torch
from datasets import DatasetDict, load_from_disk, Dataset, concatenate_datasets, load_dataset
from evaluate import load
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2FeatureExtractor, PreTrainedModel, Wav2Vec2Processor
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
from transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm import Wav2Vec2ProcessorWithLM

from src.models.config.eval_config import EvalConfig
from src.models.config.train_config import CorpusConfig
from src.train_logger import TrainLogger
from src.utils.metrics import bleu_score as bleu_metric
from src.utils.metrics import chrf_score as chrf_metric

wer_metric = load("wer")
cer_metric = load("cer")

_DEFAULT_ENV = "development"


class EvaluationPipeline:
    """
    Evaluation Pipline using Huggingface Datasets
    """

    env: str
    logger: TrainLogger

    config: EvalConfig

    logging_path: str
    model_path: str

    vocab_file: str
    processor: Union[Wav2Vec2Processor, Wav2Vec2ProcessorWithLM]
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer
    model: PreTrainedModel

    label_feature: str = "text_label"
    raw_label_feature: str = "raw_text_label"

    def __init__(self, config: EvalConfig, env: str = "production"):
        """
        Initialize the EvaluationPipeline with the given configuration file.

        :param config: The config used for this evaluation.
        :param env: Name of the environment
        """

        self.config = config
        self.env = env
        random.seed(self.config.seed)

        # Define data dir
        self.cache_dir = str(os.path.join(self.config.alt_base_path, ".cache"))
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Setup model paths
        self.model_path = str(os.path.join(config.alt_base_path, "models", config.checkpoint))
        self.vocab_file = os.path.join(config.alt_base_path, "models", config.checkpoint, '..', 'vocab.json')

        print(self.model_path)
        print(self.vocab_file)

        # Setup logging
        self.logging_path = str(os.path.join(self.config.alt_base_path, 'logs', "runs", config.group, config.job_type, config.experiment_label))
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path, exist_ok=True)
        self.logger = TrainLogger(self.logging_path)
        self.config.to_json(self.logging_path)

        self.lm_processor = None

    def run(self):
        """
        Runs the evaluation
        """

        dataset = self.load_data(self.config.test_corpora)
        if self.config.test_num_samples:
            dataset = dataset.select(range(self.config.test_num_samples))

        self.tokenizer = self.create_tokenizer()

        self.processor = self.create_processor(lm_path=self._load_language_model())
        self.model = self._get_model()

        dataset = self.prepare_dataset(dataset)

        if self.config.wandb_offline:
            self.evaluate(dataset, self.processor, self.model)
        else:
            import wandb
            with (wandb.init(
                    settings=wandb.Settings(job_name='evaluation_job'),
                    job_type=self.config.job_type,
                    group=self.config.group,
                    config=self.config.dict(),
                    name=self.config.experiment_label) as run):

                results = self.evaluate(dataset, self.processor, self.model)
                for key, value in results.items():
                    run.summary[key] = value

                # alignment_file_path = os.path.join(self.logging_path, 'alignment_result.json')
                # if os.path.exists(alignment_file_path):
                #     alignment_artifact = wandb.Artifact("alignment_results", type="json")
                #     alignment_artifact.add_file(alignment_file_path)
                #     run.log_artifact(alignment_artifact)

    def load_data(self, test_corpora: List[CorpusConfig]) -> Dataset:
        """
        Load the test dataset using Huggingface datasets.

        @param test_corpora: Config to load the test
        """

        cache_dir = os.path.join(self.cache_dir, "dataset")
        test_datasets = []

        for corpus_config in test_corpora:
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
                corpus_config.text_column: self.label_feature,
                corpus_config.raw_text_column: self.raw_label_feature
            })
            ds = ds.select_columns(
                [col for col in ["audio_id", "audio", self.label_feature, self.raw_label_feature, "duration"] if col in ds.column_names])

            # Add dataset to the list
            if isinstance(ds, DatasetDict):
                test_datasets.append(ds["test"])
            else:
                test_datasets.append(ds)

        # Concat dataset if more than one
        if len(test_datasets) > 1:
            dataset = concatenate_datasets(test_datasets)
        else:
            dataset = test_datasets[0]

        return dataset

    def create_tokenizer(self, unk_token: str = "[UNK]", pad_token: str = "[PAD]", word_delimiter_token: str = "|") -> Wav2Vec2CTCTokenizer:
        """
        Create tokenizer from saved model.

        :param unk_token: Token for unknown words.
        :param pad_token: Token for padding.
        :param word_delimiter_token: Token for word delimiter.
        """
        self.logger.log_event("Create Tokenizer", unk_token=unk_token, pad_token=pad_token, word_delimiter_token=word_delimiter_token)
        assert os.path.exists(self.vocab_file), f"Vocabulary file not found at {self.vocab_file}"

        decoder_tokenizer = Wav2Vec2CTCTokenizer(self.vocab_file, unk_token=unk_token, pad_token=pad_token, word_delimiter_token=word_delimiter_token)
        # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_path)
        # tokenizer.add_special_tokens({'pad_token': pad_token, 'unk_token': unk_token})
        decoder_tokenizer.add_special_tokens({'pad_token': pad_token, 'unk_token': unk_token})
        return decoder_tokenizer

    def create_processor(self, feature_size: int = 1, sampling_rate: int = 16_000, lm_path: str = None) -> Wav2Vec2Processor:
        """
        Create processor from the saved Wav2Vec2 model.

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

        if self.config.eval_beam_size > 1:
            vocab_dict = self.tokenizer.get_vocab()
            sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
            sorted_dict_keys = list(sorted_dict.keys())
            # sorted_dict_keys[vocab_dict['|']] = ' '  # ugly hack to make delimiter from | to  ' '
            # sorted_dict_keys[vocab_dict[' ']] = '|'  # ugly hack to make delimiter from | to  ' '

            language_model_decoder = build_ctcdecoder(
                labels=sorted_dict_keys,
                kenlm_model_path=lm_path,
                alpha=self.config.lm_alpha,
                beta=self.config.lm_beta
            )

            self.lm_processor = Wav2Vec2ProcessorWithLM(
                tokenizer=self.tokenizer,
                feature_extractor=feature_extractor,
                decoder=language_model_decoder
            )

        return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=self.tokenizer)

    def _get_model(self) -> Wav2Vec2ForCTC:
        """
        Load the Wav2Vec2 model.

        :return: Wav2Vec2ForCTC model.
        """

        assert self.processor, "No processor"
        self.logger.log_event("Load Model", model_path=self.model_path)

        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_path,
            # attention_dropout=0.1,
            # hidden_dropout=0.1,
            # feat_proj_dropout=0.0,
            # mask_time_prob=0.05,
            # layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer)
        )
        model.config.ctc_zero_infinity = True
        model.freeze_feature_encoder()
        model.to(self.config.device)
        return model

    def _load_language_model(self):
        """
        Load an optional language model.

        :return: Path to the language model.
        """

        lm_path = None
        if self.config.kenlm_model_name is not None:
            try:
                import kenlm
                lm_path = os.path.join(self.config.alt_base_path, 'language_modelling', config['kenlm_model_name'])
                lm_path_test = os.path.join(self.config.alt_base_path, 'language_modelling', 'test.arpa')
                kenlm.Model(lm_path_test)
            except Exception as _:
                print('Cannot Load KenLM')
                lm_path = None

        return lm_path

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare the dataset for testing.

        @param dataset: The test dataset to be prepared.
        @return: The tokenized dataset.
        """

        self.logger.log_event("Prepare Data")
        remove_columns = dataset.column_names

        # Add filtering to only keep samples with at least 2 words
        def filter_by_word_count(batch):
            return len(batch[self.raw_label_feature].split()) >= self.config.min_word_count

        # Apply filtering
        dataset = dataset.filter(filter_by_word_count, num_proc=1)

        def _prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch[self.label_feature]).input_ids
            batch["raw_labels"] = batch[self.raw_label_feature]  # keep raw text as well
            return batch

        dataset = dataset.map(_prepare_dataset, remove_columns=remove_columns, num_proc=1, load_from_cache_file=True)
        self.logger.log_event("Data Prepared")
        return dataset

    def evaluate(self, dataset, processor, model):
        """
        Evaluates the given dataset using the specified model and processor, calculates evaluation metrics,
        and logs the results to a text file and JSON file.

        @param dataset: The dataset to evaluate. It should include input values, labels, and raw labels.
        @param processor: The processor used to decode predictions and labels.
        @param model: The model used to generate predictions from the input values.
        @return: The tokenized dataset.
        """

        log_file_path = os.path.join(self.logging_path, 'evaluation.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write("Starting new evaluation...\n")

        def log_to_file(message):
            print(message)
            with open(log_file_path, 'a') as log_file:
                log_file.write(message + '\n')

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

        def map_to_result_batch(batch):
            with torch.no_grad():

                prepared_batch = data_collator(
                    [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
                )

                # Move input values to the appropriate device
                input_values = prepared_batch["input_values"].to(self.config.device)
                logits = model(input_values).logits

                if self.config.eval_beam_size > 1 and self.lm_processor:
                    output = self.lm_processor.batch_decode(
                        logits.double().cpu().numpy(),
                        beam_width=self.config.eval_beam_size,
                        num_processes=1,
                        output_word_offsets=True
                    )
                    pred_str_batch = output.text
                else:
                    pred_ids = torch.argmax(logits, dim=-1)
                    pred_str_batch = processor.batch_decode(pred_ids, group_tokens=True, skip_special_tokens=False)

                # Decode predictions and labels
                batch["pred_str"] = pred_str_batch
                batch["label_str"] = processor.batch_decode(prepared_batch["labels"], group_tokens=False, skip_special_tokens=True)
                batch["raw_label_str"] = batch.get("raw_labels", "")
            return batch

        def map_to_result(batch):
            with torch.no_grad():
                input_values = torch.tensor(batch["input_values"], device=self.config.device).unsqueeze(0)
                logits = model(input_values).logits

                if self.config.eval_beam_size > 1 and self.lm_processor:
                    output = self.lm_processor.batch_decode(
                        logits.double().cpu().numpy(),
                        beam_width=self.config.eval_beam_size,
                        num_processes=1,
                        output_word_offsets=True
                    )

                    pred_str_batch = output.text[0]
                else:
                    pred_ids = torch.argmax(logits, dim=-1)
                    pred_str_batch = processor.batch_decode(pred_ids, group_tokens=True, skip_special_tokens=False)[0]

                # Decode predictions and labels
                batch["pred_str"] = pred_str_batch
                batch["label_str"] = processor.decode(batch["labels"], group_tokens=False, skip_special_tokens=True)
                batch["raw_label_str"] = batch.get("raw_labels", "")
            return batch

        if self.config.eval_batch_size > 1:
            results = dataset.map(map_to_result_batch, remove_columns=dataset.column_names,
                                  batch_size=self.config.eval_batch_size, batched=True)
        else:
            results = dataset.map(map_to_result, remove_columns=dataset.column_names)

        # Calculate standard metrics
        wer_score = wer_metric.compute(predictions=results["pred_str"], references=results["label_str"])
        cer_score = cer_metric.compute(predictions=results["pred_str"], references=results["label_str"])
        bleu_score = bleu_metric(predictions=results["pred_str"], references=results["label_str"])
        chrf_score = chrf_metric(predictions=results["pred_str"], references=results["label_str"])

        # Log standard metrics
        log_to_file("Test WER: {:.3f}".format(wer_score))
        log_to_file("Test CER: {:.3f}".format(cer_score))
        log_to_file("Test Bleu: {:.3f}".format(bleu_score))
        log_to_file("Test ChrF: {:.3f}".format(chrf_score))

        # Save predictions and labels to a CSV file
        with open(os.path.join(self.logging_path, 'predictions.csv'), 'w') as csv_file:
            import csv
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Raw Label", "Label", "Prediction"])
            for raw, pred, label in zip(results["raw_label_str"], results["label_str"], results["pred_str"]):
                csv_writer.writerow([raw, pred, label])

        wepr_result_dict = {}
        if config.wepr:
            # Calculate Error Preservation
            annot_eval = AnnotationEvaluation(references=results["raw_label_str"], list_of_predictions=[results["pred_str"]], symbols=["@!"],
                                              num_threads=5)
            wepr_score = annot_eval.calculate_aer()  # todo is this correct. calculate few examples by hand!!!

            # Log error preservation metrics and infos
            log_to_file("Test WEPR: {:.3f}".format(wepr_score))
            log_to_file("Correct Words: {}".format(annot_eval.correct_words))
            log_to_file("Substitutions: {}".format(annot_eval.substitutions))
            log_to_file("Insertions: {}".format(annot_eval.insertions))
            log_to_file("Deletions: {}".format(annot_eval.deletions))
            log_to_file("Reference Words: {}".format(annot_eval.reference_words))

            # Save alignment to JSON file
            with open(os.path.join(self.logging_path, 'alignment_result.json'), 'w') as json_file:
                json.dump(annot_eval.alignment_result, json_file)

            wepr_result_dict = {
                'wepr': wepr_score,
                'correct_words': annot_eval.correct_words,
                'substitutions': annot_eval.substitutions,
                'insertions': annot_eval.insertions,
                'deletions': annot_eval.deletions,
                'reference_words': annot_eval.reference_words,
            }

        return {
            'bleu': bleu_score,
            'wer': wer_score,
            'cer': cer_score,
            'chrf': chrf_score,
            **wepr_result_dict
        }


if __name__ == '__main__':
    # os.environ["WANDB_MODE"] = "offline"
    env = os.environ.get('ENV', _DEFAULT_ENV)

    if env == 'development':
        from dotenv import load_dotenv

        load_dotenv()
        print("Load .env")

    # Combine configs to use defaults and experiment-specific configs
    config = EvalConfig.from_cli()
    print(config.model_dump())

    pipeline = EvaluationPipeline(config=config, env=env)
    pipeline.run()

import collections
import json
import os
import random
from typing import List, Union

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
        self.vocab_file = os.path.join(config.alt_base_path, "models", config.checkpoint, 'vocab.json')

        # Setup logging
        self.logging_path = str(os.path.join(self.config.alt_base_path, 'logs', "runs", config.group, config.job_type, config.experiment_label))
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path, exist_ok=True)
        self.logger = TrainLogger(self.logging_path)
        self.config.to_json(self.logging_path)

    def run(self, log_wandb: bool = False):
        """
        Runs the evaluation
        """

        dataset = self.load_data(self.config.test_corpora)
        # dataset = dataset.select(range(10))

        self.tokenizer = self.create_tokenizer()

        self.processor = self.create_processor(lm_path=self._load_language_model())
        self.model = self._get_model()

        dataset = self.prepare_dataset(dataset)

        if not log_wandb:
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

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_path)
        tokenizer.add_special_tokens({'pad_token': pad_token, 'unk_token': unk_token})
        return tokenizer

    def create_processor(self, feature_size: int = 1, sampling_rate: int = 16_000, lm_path: str = None) \
            -> Union[Wav2Vec2Processor, Wav2Vec2ProcessorWithLM]:
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

        if lm_path:
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

            return Wav2Vec2ProcessorWithLM(
                tokenizer=self.tokenizer,
                feature_extractor=feature_extractor,
                decoder=language_model_decoder
            )
        else:
            return Wav2Vec2Processor.from_pretrained(self.model_path, tokenizer=self.tokenizer)

    def _get_model(self) -> Wav2Vec2ForCTC:
        """
        Load the Wav2Vec2 model.

        :return: Wav2Vec2ForCTC model.
        """

        assert self.processor, "No processor"
        self.logger.log_event("Load Model", model_path=self.model_path)

        model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
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

        def map_to_result(batch):
            with torch.no_grad():
                input_values = torch.tensor(batch["input_values"], device=self.config.device).unsqueeze(0)
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            batch["label_str"] = processor.decode(batch["labels"], group_tokens=False)
            batch["raw_label_str"] = batch["raw_labels"]
            return batch

        results = dataset.map(map_to_result, remove_columns=dataset.column_names)

        # Calculate Error Preservation
        annot_eval = AnnotationEvaluation(references=results["raw_label_str"], list_of_predictions=[results["pred_str"]], symbols=["@!"])
        wepr_score = annot_eval.calculate_aer()

        # todo is this correct. calculate few examples by hand!!!

        # Calculate standard metrics
        wer_score = wer_metric.compute(predictions=results["pred_str"], references=results["label_str"])
        cer_score = cer_metric.compute(predictions=results["pred_str"], references=results["label_str"])
        bleu_score = bleu_metric(predictions=results["pred_str"], references=results["label_str"])
        chrf_score = chrf_metric(predictions=results["pred_str"], references=results["label_str"])

        # Log metrics to file
        log_to_file("Test WER: {:.3f}".format(wer_score))
        log_to_file("Test CER: {:.3f}".format(cer_score))
        log_to_file("Test Bleu: {:.3f}".format(bleu_score))
        log_to_file("Test ChrF: {:.3f}".format(chrf_score))
        log_to_file("Test WEPR: {:.3f}".format(wepr_score))

        log_to_file("Correct Words: {}".format(annot_eval.correct_words))
        log_to_file("Substitutions: {}".format(annot_eval.substitutions))
        log_to_file("Insertions: {}".format(annot_eval.insertions))
        log_to_file("Deletions: {}".format(annot_eval.deletions))
        log_to_file("Reference Words: {}".format(annot_eval.reference_words))

        # Save alignment to JSON file
        with open(os.path.join(self.logging_path, 'alignment_result.json'), 'w') as json_file:
            json.dump(annot_eval.alignment_result, json_file)

        return {
            'bleu': bleu_score,
            'wer': wer_score,
            'cer': cer_score,
            'chrf': chrf_score,
            'wepr': wepr_score,
            'correct_words': annot_eval.correct_words,
            'substitutions': annot_eval.substitutions,
            'insertions': annot_eval.insertions,
            'deletions': annot_eval.deletions,
            'reference_words': annot_eval.reference_words,
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
    pipeline = EvaluationPipeline(config=config, env=env)
    pipeline.run(log_wandb=True)


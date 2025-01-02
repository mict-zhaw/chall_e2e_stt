import math
import os
import random
from collections import defaultdict
from typing import List, Union

import torch
from datasets import DatasetDict, load_from_disk, Dataset, concatenate_datasets, load_dataset
from evaluate import load
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, PreTrainedModel, Wav2Vec2Processor
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
from transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm import Wav2Vec2ProcessorWithLM

from src.models.config.eval_config import EvalConfig
from src.models.config.train_config import CorpusConfig
from src.train_logger import TrainLogger
from src.utils.bleu_metric import score as bleu_metric

wer_metric = load("wer")
cer_metric = load("cer")
ter_metric = load("ter")

_DEFAULT_ENV = "production"


class EvaluationPipeline:
    """

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

    def __init__(self, config: EvalConfig, env: str = "production"):
        """

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
        self.vocab_file = os.path.join(config.alt_base_path, "models", config.checkpoint, "..", 'vocab.json')

        # Setup logging
        self.logging_path = str(os.path.join(self.config.alt_base_path, 'logs', "runs", config.group, config.job_type, config.experiment_label))
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path, exist_ok=True)
        self.logger = TrainLogger(self.logging_path)
        self.config.to_json(self.logging_path)

    def run(self):

        dataset = self.load_data(self.config.test_corpora)
        dataset = dataset.select(range(10)) # todo

        self.tokenizer = self.create_tokenizer()

        self.processor = self.create_processor(lm_path=self._load_language_model())
        self.model = self._get_model()

        dataset = self.prepare_dataset(dataset)
        self.evaluate(dataset, self.processor, self.model)

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
                corpus_config.text_column: self.label_feature
            })
            ds = ds.select_columns([col for col in ["audio_id", "audio", self.label_feature, "duration"] if col in ds.column_names])

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
        Create a tokenizer for the Wav2Vec2 model.

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

        vocab_dict = self.tokenizer.get_vocab()
        sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        sorted_dict_keys = list(sorted_dict.keys())
        # sorted_dict_keys[vocab_dict['|']] = ' '  # ugly hack to make delimiter from | to  ' '
        # sorted_dict_keys[vocab_dict[' ']] = '|'  # ugly hack to make delimiter from | to  ' '

        if lm_path:
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
            test = Wav2Vec2Processor.from_pretrained(self.model_path, tokenizer=self.tokenizer)
            return test


    def _get_model(self) -> Wav2Vec2ForCTC:
        """
        Load or initialize the Wav2Vec2 model.

        :return: Wav2Vec2ForCTC model.
        """

        assert self.processor, "No processor"

        self.logger.log_event("Load Model", model_path=self.model_path)

        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_path,
            # cache_dir=os.path.join(self.cache_dir, "models"),
            # attention_dropout=0.1,
            # hidden_dropout=0.1,
            # feat_proj_dropout=0.0,
            # mask_time_prob=0.05,
            # layerdrop=0.1,
            # ctc_loss_reduction="mean",
            # pad_token_id=self.processor.tokenizer.pad_token_id,
            # vocab_size=len(self.tokenizer),
        )

        model.config.ctc_zero_infinity = True
        model.freeze_feature_encoder()
        model.to(self.config.device)
        return model

    def _load_language_model(self):

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

    @property
    def dtype(self):
        if self.config.bf16:
            return torch.bfloat16
        elif self.config.fp16:
            return torch.float16
        else:
            return torch.float

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare the dataset for testing.
        """

        self.logger.log_event("Prepare Data")
        remove_columns = dataset.column_names

        def _prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch[self.label_feature]).input_ids
            return batch

        dataset = dataset.map(_prepare_dataset, remove_columns=remove_columns, num_proc=1, load_from_cache_file=True)
        self.logger.log_event("Data Prepared")
        return dataset

    def evaluate(self, dataset, processor, model):

        def map_to_result(batch):
            with torch.no_grad():
                input_values = torch.tensor(batch["input_values"], device=self.config.device).unsqueeze(0)
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            batch["label_str"] = processor.decode(batch["labels"], group_tokens=False)

            return batch

        results = dataset.map(map_to_result, remove_columns=dataset.column_names)

        # calculate scores
        wer_scores = wer_metric.compute(predictions=results["pred_str"], references=results["label_str"])
        bleu_scores = bleu_metric(predictions=results["pred_str"], references=results["label_str"])

        print("Test WER: {:.3f}".format(wer_scores))
        print("Test Bleu: {:.3f}".format(bleu_scores))

        # todo add BLEU and WEPR---> this is going to be fun...

    def on_evaluate(self, epoch):
        """
        todo delete
        """

        self.model.eval()
        n_eval_samples = len(test_sample_list)
        total_steps = math.ceil(n_eval_samples / self.config.eval_batch_size)
        full_files, final_files = {}, {}
        for corpus in test_corpora_list:
            corpus_name, dataset_name = corpus[0], corpus[1]
            logging_out_path = os.path.join('logging2', experiment_name, experiment_tag, 'outputs', checkpoint,
                                            f'{corpus_name}-{dataset_name}-beam{eval_beam_size}')
            if not os.path.exists(logging_out_path):
                os.makedirs(logging_out_path)
            full_files[corpus_name] = open(os.path.join(logging_out_path, f'output_full_{epoch}.txt'), 'wt', encoding='utf-8')
            final_files[corpus_name] = open(os.path.join(logging_out_path, f'output_{epoch}.txt'), 'wt', encoding='utf-8')
            full_files[corpus_name].write('Dummy First Line to Maintain Consistency\n')

        pred_strs, label_strs = defaultdict(lambda: []), defaultdict(lambda: [])
        sample_names = defaultdict(lambda: [])
        for step in tqdm(range(0, n_eval_samples, eval_batch_size), desc='Inference', total=total_steps):
            start_idx = step
            end_idx = step + eval_batch_size
            test_batch = test_sample_list[start_idx:end_idx]
            batch_sample_names = [(el['input_values'].corpus_name,
                                   el['input_values'].dataset_name,
                                   el['input_values'].audio_id.upper()) for el in test_batch]

            for corpus_name, dataset_name, sample_name in batch_sample_names:
                sample_names[corpus_name].append(sample_name)

            inputs = collator(test_batch)

            label_str = decoder_tokenizer.batch_decode(inputs['labels'], group_tokens=False, skip_special_tokens=True)

            with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
                logits = model(inputs['input_values'].to(device)).logits
                if eval_beam_size > 1:
                    output = wav2vec_lm_processor.batch_decode(
                        logits.double().cpu().numpy(),
                        beam_width=eval_beam_size,
                        num_processes=1,
                        output_word_offsets=True
                    )

                    pred_str_batch = output.text
                else:
                    pred_idx = torch.argmax(logits, dim=-1)
                    pred_str_batch = decoder_tokenizer.batch_decode(pred_idx, group_tokens=True, skip_special_tokens=True)

            for tok in special_tokens_print:
                if tok is not None:
                    pred_str_batch = [x.replace(tok, '') for x in pred_str_batch]
                    label_str = [x.replace(tok, '').strip() for x in label_str]

            for (corpus_name, _, _), pred, ref, r in zip(batch_sample_names, pred_str_batch, label_str, test_batch):
                pred_strs[corpus_name].append(pred)
                label_strs[corpus_name].append(ref)
                r["input_values"].hypothesis = pred

        for corpus_name, ofile in final_files.items():

            valid_pairs = [(p, r) for p, r in zip(pred_strs[corpus_name], label_strs[corpus_name]) if p and r]
            valid_predictions, valid_references = zip(*valid_pairs) if valid_pairs else ([], [])

            print("len red_strs[corpus_name]", len(pred_strs[corpus_name]), ", valid_predictions", len(valid_predictions))

            try:
                wer_score = wer_metric.compute(predictions=valid_predictions, references=valid_references)
                bleu_score = score(pred_strs[corpus_name], label_strs[corpus_name])

            except ValueError as e:
                print("Cannot compute WER: one or more references are empty strings")
                wer_score = None
                bleu_score = None

            ofile.write(f'WER: {wer_score}\tBLEU: {bleu_score}\n')
            for idx, (pred_str, label_str, sample_name) in enumerate(zip(pred_strs[corpus_name], label_strs[corpus_name], sample_names[corpus_name])):
                ofile.write(f'{idx}\t{sample_name}\t{pred_str}\t{label_str}\n')


if __name__ == '__main__':
    # os.environ["WANDB_MODE"] = "offline"
    env = os.environ.get('ENV', _DEFAULT_ENV)

    # Combine configs to use defaults and experiment-specific configs
    # config = TrainConfig.from_yaml(args.default_config, args.config)
    config = EvalConfig.from_cli(default_config_file="config/eval/eval_debug.yaml")
    pipeline = EvaluationPipeline(config=config, env=env)
    pipeline.run()

    # todo how to calculate WPER?

import collections
import json
import os
import random
import string
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

        print(self.model_path)
        print(self.vocab_file)

        # Setup logging
        self.logging_path = str(os.path.join(self.config.alt_base_path, 'logs', "runs", config.group, config.job_type, config.experiment_label))
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path, exist_ok=True)
        self.logger = TrainLogger(self.logging_path)
        self.config.to_json(self.logging_path)

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
            # ds = ds.select_columns(
            #     [col for col in ["audio_id", "audio", self.label_feature, self.raw_label_feature, "duration"] if col in ds.column_names])

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

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_path)
        tokenizer.add_special_tokens({'pad_token': pad_token, 'unk_token': unk_token})
        decoder_tokenizer.add_special_tokens({'pad_token': pad_token, 'unk_token': unk_token})
        return decoder_tokenizer

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
        remove_columns = ["audio"]

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

        results = dataset.map(map_to_result) # remove_columns=dataset.column_names

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

        # todo evaluate ERRANT...

        self.errant_analysis(results)

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

    def _classify_spans(self, original_spans: set, corrected_spans: set):
        hits = original_spans.intersection(corrected_spans)  # Matched errors
        misses = original_spans.difference(corrected_spans)  # Unmatched errors
        unnecessary = corrected_spans.difference(original_spans)  # Extra generated errors

        return hits, misses, unnecessary

    def _calculate_scores(self, hits, misses, unnecessary):
        precision = hits / (hits + unnecessary) if (hits + unnecessary) > 0 else 0
        recall = hits / (hits + misses) if (hits + misses) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def _grammatical_error_correction(self, input_sentence, num_return_sequences=5):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        # Load pre-trained model and tokenizer
        model_name = "vennify/t5-base-grammar-correction"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Tokenize input
        input_ids = tokenizer(f"grammar: {input_sentence}", return_tensors="pt").input_ids

        # Generate multiple corrected versions
        outputs = model.generate(
            input_ids,
            max_length=50,
            num_beams=num_return_sequences,  # Beam search for diverse outputs
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Decode and calculate probabilities
        corrected_sentences = []
        scores = outputs.sequences_scores  # Log probabilities for each sequence

        for i, output in enumerate(outputs.sequences):
            corrected_text = tokenizer.decode(output, skip_special_tokens=True)
            log_prob = scores[i].item()
            corrected_sentences.append({"text": corrected_text, "log_prob": log_prob})

        # print(corrected_sentences)

        # Normalize probabilities
        total_prob = sum(torch.exp(torch.tensor([item["log_prob"] for item in corrected_sentences])))
        for item in corrected_sentences:
            item["probability"] = torch.exp(torch.tensor(item["log_prob"])) / total_prob

        return corrected_sentences

        # Calculate mean probability
        mean_probability = sum(item["probability"] for item in corrected_sentences) / len(corrected_sentences)

        # Filter results with probability >= mean
        filtered_sentences = [item for item in corrected_sentences if item["probability"] >= mean_probability]

        # Display filtered results
        print(f"Mean Probability: {mean_probability:.4f}")
        for idx, correction in enumerate(filtered_sentences):
            print(f"Version {idx + 1}: {correction['text']} (Probability: {correction['probability']:.4f})")

        return filtered_sentences

    def _calculate(self, original_spans: set, corrected_spans: set):
        hits, misses, unnecessary = self._classify_spans(original_spans, corrected_spans)
        return self._calculate_scores(len(hits), len(misses), len(unnecessary))

    def errant_analysis(self, rows: Dataset, text_key: str = "pred_str", test_text_key: str = "raw_label_str"):

        import errant

        file_path = '../../data/raw/annotated_samples_uzh/8400_4.jsonl'

        print(rows)

        annotator = errant.load('en')

        result = []

        orig_hits = []
        orig_missing = []
        orig_unnecessary = []

        corr_hits = []
        corr_missing = []
        corr_unnecessary = []

        for row in rows:

            # Create corrections for the generated incorrect text
            o_text = row[text_key]
            corrections = self._grammatical_error_correction(o_text, num_return_sequences=3)

            print("Ref:", row["raw_label_str"])
            print("Ref:", row["label_str"])
            print("Input:", o_text)
            # print("Orig Correction:", row["corrected_text"])
            print("Corrections:", [c["text"] for c in corrections])
            print("Input Errors:", set(row["uzh_errant_errors"]))
            # print("Pair Errors:", [e["label"] for e in row["spans"]])

            if row["label_str"].lower() == row[text_key].lower():
                print("same same")
                continue

            if len(set(row["uzh_errant_errors"])) == 0:
                print("no errors")
                continue

            o_text = annotator.parse(o_text.translate(str.maketrans('', '', string.punctuation)).lower(), tokenise=True)

            # Get a list of generated and original errors
            generated_spans = set(row["uzh_errant_errors"])
            # original_spans = set([e["label"] for e in row["spans"]])

            # todo generated spans sind hier diejenigen von UZH... wie lade ich die?

            # Calculate Scores between the correct and incorrect generated sentence pair
            # o_hits, o_misses, o_unnecessary = self._classify_spans(generated_spans, original_spans)
            # orig_hits += list(o_hits)
            # orig_missing += list(o_misses)
            # orig_unnecessary += list(o_unnecessary)
            # original_scores = self._calculate(generated_spans, original_spans)

            # Find the best correction in terms of error preservation
            best_hits = []
            best_missing = []
            best_unnecessary = []
            best_spans = []
            best_scores = {"f1_score": 0, "precision": 0, "recall": 0}
            for correction in corrections:
                c_text = annotator.parse(correction["text"].translate(str.maketrans('', '', string.punctuation)).lower(), tokenise=True)


                print(o_text, "<->", c_text)
                # Annotate errors between correction and original generated incorrect sentence
                error_annotations = annotator.annotate(o_text, c_text)
                correction_spans = set([e.type for e in error_annotations])

                # Classify the annotations
                c_hits, c_misses, c_unnecessary = self._classify_spans(generated_spans, correction_spans)
                c_scores = self._calculate_scores(len(c_hits), len(c_misses), len(c_unnecessary))

                # Update for new best scores
                if c_scores["recall"] >= best_scores["recall"]:
                    best_scores = c_scores
                    best_spans = correction_spans
                    best_hits = list(c_hits)
                    best_missing = list(c_misses)
                    best_unnecessary = list(c_unnecessary)

            # Update lists with classified errors
            corr_hits += best_hits
            corr_missing += best_missing
            corr_unnecessary += best_unnecessary

            print("Hits:", best_hits)
            print("Miss:", best_missing)
            print("Unn:", best_unnecessary)
            if len(set(row["uzh_errant_errors"])) > 0:
                print("Precision Hits:", len(best_hits)/len(set(row["uzh_errant_errors"])))
            else:
                print("no errors")

            print("Correction Errors: ", best_spans)
            # print(original_scores)
            print(best_scores)
            print("---------------")

        o_scores = self._calculate_scores(len(orig_hits), len(orig_missing), len(orig_unnecessary))
        c_scores = self._calculate_scores(len(corr_hits), len(corr_missing), len(corr_unnecessary))

        print("Scores between incorrect and generated correct sentence:", o_scores)
        print("Scores between incorrect and GEC corrected sentence:", c_scores)


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

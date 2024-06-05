from typing import Union, TypeVar
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf
from evaluate import load

wer = load("wer")
cer = load("cer")
ter = load("ter")

T = TypeVar('T', bound='BaseTranscriptPreprocessor')


def prepare(predictions: Union[str, list[str]], references: Union[str, list[str]], preprocessor: T = None, remove_empty_references: bool = False, split: bool = False):
    """

    :param predictions:
    :param references:
    :param preprocessor:
    :param remove_empty_references:
    :return:
    """

    if isinstance(predictions, str):
        predictions = [predictions]

    if isinstance(references, str):
        references = [references]

    if preprocessor:
        _preprocessor = type(preprocessor)()
        references = [_preprocessor.preprocess(reference) for reference in references]
        predictions = [_preprocessor.preprocess(prediction) for prediction in predictions]

    if remove_empty_references:
        references, predictions = zip(*[(ref, pred) for ref, pred in zip(references, predictions) if ref != ""])

    if split:
        references = [reference.split(' ') for reference in references]
        predictions = [prediction.split(' ') for prediction in predictions]

    return references, predictions


def bleu_score(predictions: Union[str, list[str]], references: Union[str, list[str]], preprocessor: T = None) -> float:
    """
    Calculate the BLEU score between a list of predicted sentences and a list of reference sentences.
    :param predictions: A list of predicted sentences.
    :param references:  A list of reference sentences.
    :param preprocessor:
    :return: The BLEU score. The score ranges from 0 to 1, where higher values indicate better translation quality.
    """

    references, predictions = prepare(predictions, references, preprocessor=preprocessor)

    list_of_references = []
    hypotheses = []
    for prediction, reference in zip(predictions, references):
        list_of_references.append([reference.split(' ')])
        hypotheses.append(prediction.split(' '))

    return corpus_bleu(list_of_references, hypotheses)


def sentence_bleu_score(prediction: str, reference: str, use_smoothing: bool = True, preprocessor: T = None) -> float:
    """
    Calculate the BLEU score between a single predicted sentence and a single reference sentence.
    :param prediction: The predicted sentence.
    :param reference: The reference sentence.
    :param use_smoothing:
    :param preprocessor:
    :return: The BLEU score. The score ranges from 0 to 1, where higher values indicate better translation quality.
    """

    references, predictions = prepare(prediction, reference, preprocessor=preprocessor)

    if use_smoothing:
        smoothing = SmoothingFunction()
        return sentence_bleu(references, predictions[0], smoothing_function=smoothing.method1)
    else:
        return sentence_bleu(references, predictions[0])


def wer_score(predictions: Union[str, list[str]], references: Union[str, list[str]], preprocessor: T = None) -> float:
    """
    Word Error Rate
    :param predictions:
    :param references:
    :param preprocessor:
    :return:
    """

    references, predictions = prepare(predictions, references, preprocessor=preprocessor, remove_empty_references=True)
    return wer.compute(predictions=predictions, references=references)


def cer_score(predictions: Union[str, list[str]], references: Union[str, list[str]], preprocessor: T = None) -> float:
    """
    Character Error Rate
    :param predictions:
    :param references:
    :param preprocessor:
    :return:
    """

    references, predictions = prepare(predictions, references, preprocessor=preprocessor, remove_empty_references=True)
    return cer.compute(predictions=predictions, references=references)


def ter_score(predictions: Union[str, list[str]], references: Union[str, list[str]], preprocessor: T = None) -> float:
    """
    Translation Error Rate
    :param predictions:
    :param references:
    :param preprocessor:
    :return:
    """
    references, predictions = prepare(predictions, references, preprocessor=preprocessor, remove_empty_references=True)
    return ter.compute(predictions=predictions, references=references, case_sensitive=False)


def chrf_score(predictions: Union[str, list[str]], references: Union[str, list[str]], preprocessor: T = None) -> float:
    """
    Character-level F-Score
    :param predictions:
    :param references:
    :param preprocessor:
    :return:
    """

    references, predictions = prepare(predictions, references, preprocessor=preprocessor)

    list_of_references = []
    hypotheses = []
    for prediction, reference in zip(predictions, references):
        list_of_references.append(reference.split(' '))
        hypotheses.append(prediction.split(' '))

    return corpus_chrf(list_of_references, hypotheses)




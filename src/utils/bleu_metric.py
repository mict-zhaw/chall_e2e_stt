import os
import re
import sys
from src.utils.preprocess_utils import split, sentence_nums_to_words
import warnings
warnings.filterwarnings('ignore')

from nltk.translate.bleu_score import corpus_bleu
import pandas as pd

ALLOWED_CHARS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'ä', 'ö', 'ü',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ' ',
}

WHITESPACE_REGEX = re.compile(r'[ \t]+')

NUMBER_REGEX = re.compile(r"^[0-9',.]+$")
NUMBER_DASH_REGEX = re.compile('[0-9]+[-\u2013\xad]')
DASH_NUMBER_REGEX = re.compile('[-\u2013\xad][0-9]+')


def preprocess_transcript(transcript):
    transcript = transcript.lower()
    transcript = transcript.replace('ß', 'ss')
    transcript = transcript.replace('ç', 'c')
    transcript = transcript.replace('á', 'a')
    transcript = transcript.replace('à', 'a')
    transcript = transcript.replace('â', 'a')
    transcript = transcript.replace('é', 'e')
    transcript = transcript.replace('è', 'e')
    transcript = transcript.replace('ê', 'e')
    transcript = transcript.replace('í', 'i')
    transcript = transcript.replace('ì', 'i')
    transcript = transcript.replace('î', 'i')
    transcript = transcript.replace('ó', 'o')
    transcript = transcript.replace('ò', 'o')
    transcript = transcript.replace('ô', 'o')
    transcript = transcript.replace('ú', 'u')
    transcript = transcript.replace('ù', 'u')
    transcript = transcript.replace('û', 'u')
    transcript = transcript.replace('-', ' ')
    transcript = transcript.replace('\u2013', ' ')
    transcript = transcript.replace('\xad', ' ')
    transcript = transcript.replace('/', ' ')
    transcript = WHITESPACE_REGEX.sub(' ', transcript)
    transcript = ''.join([char for char in transcript if char in ALLOWED_CHARS])
    transcript = WHITESPACE_REGEX.sub(' ', transcript)
    transcript = transcript.strip()

    return transcript


def score(predictions, references):
    list_of_references = []
    hypotheses = []
    for prediction, reference in zip(predictions, references):

        true_sentence_nums_to_words = sentence_nums_to_words(reference)
        list_of_references.append([
            split(preprocess_transcript(reference)),
            split(preprocess_transcript(true_sentence_nums_to_words)),
        ])
        hypotheses.append(split(preprocess_transcript(prediction)))

    return corpus_bleu(list_of_references, hypotheses)


if __name__ == '__main__':
    path_to_script = os.path.dirname(sys.argv[0])
    df_submission = pd.read_csv(os.path.join(path_to_script, 'submission.csv'), sep=',', encoding='utf-8')
    df_public = pd.read_csv(os.path.join(path_to_script, 'public.csv'), sep=',', encoding='utf-8')
    df_private = pd.read_csv(os.path.join(path_to_script, 'private.csv'), sep=',', encoding='utf-8')

    # df_submission = pd.read_csv(os.path.join('data', 'example_submission.csv'), sep=',', encoding='utf-8')
    # df_public = pd.read_csv(os.path.join('data', 'test_set_public.csv'), sep=',', encoding='utf-8')
    # df_private = pd.read_csv(os.path.join('data', 'test_set_private.csv'), sep=',', encoding='utf-8')

    score_public = score(df_public, df_submission)
    score_private = score(df_private, df_submission)

    print('%.20f' % score_public, ';', '%.20f' % score_private)
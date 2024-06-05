import re
from decimal import DecimalException
from num2words import num2words
WHITESPACE_REGEX = re.compile(r'[ \t]+')

NUMBER_REGEX = re.compile(r"^[0-9',.]+$")
NUMBER_DASH_REGEX = re.compile('[0-9]+[-\u2013\xad]')
DASH_NUMBER_REGEX = re.compile('[-\u2013\xad][0-9]+')


def split(transcript):
    return transcript.split(' ')


def sentence_nums_to_words(transcript):
    def transform_word(word):
        # num2words if the word is just one number
        if NUMBER_REGEX.match(word) is not None:
            try:
                if word.endswith('.'):
                    return num2words(word, lang='de', to='ordinal')
                else:
                    return num2words(word, lang='de', to='cardinal')
            except DecimalException:
                return word
        else:
            # num2words for the number part if the word contains a number followed by a dash
            match = NUMBER_DASH_REGEX.search(word)
            if match is not None:
                num = word[match.start():match.end() - 1]
                try:
                    num = num2words(num, lang='de', to='cardinal')
                except DecimalException:
                    pass
                word = word[:match.start()] + num + word[match.end() - 1:]

            # num2words for the number part if the word contains a dash followed by a number
            match = DASH_NUMBER_REGEX.search(word)
            if match is not None:
                num = word[match.start() + 1:match.end()]
                try:
                    num = num2words(num, lang='de', to='cardinal')
                except DecimalException:
                    pass
                word = word[:match.start() + 1] + num + word[match.end():]

            return word

    transcript = WHITESPACE_REGEX.sub(' ', transcript)
    transcript = transcript.strip()

    return ' '.join(transform_word(w) for w in split(transcript))

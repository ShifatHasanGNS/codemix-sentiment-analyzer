"""
Shared text-cleaning utilities used before tokenization/feature extraction.

TODO:
- clean_text(text: str) -> str
    Light cleaning (strip extra whitespace, normalize quotes, optionally
    lowercase Latin-script substrings while leaving Bangla script untouched).
- remove_stopwords(tokens: list[str], language: str) -> list[str]
    Optional stopword removal; needs a Bangla + English stopword list
    (NLTK provides English; Bangla stopwords may need a small hardcoded
    list or an external resource).
- load_split(split_name: str) -> pandas.DataFrame
    Convenience loader for data/processed/{train,val,test}.csv.
"""


def clean_text(text: str) -> str:
    raise NotImplementedError


def remove_stopwords(tokens: list, language: str) -> list:
    raise NotImplementedError


def load_split(split_name: str):
    raise NotImplementedError

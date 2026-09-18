"""
Manually implemented tokenizer for mixed Bangla-English vocabulary.

Must handle all 4 language conditions reasonably:
    - English: standard word tokenization
    - Bangla (Bangla script): script-aware tokenization (not whitespace-only,
      since Bangla punctuation/word-boundary conventions differ from English)
    - Banglish (Bangla written in Roman script): word tokenization, aware
      that these tokens will NOT match an English dictionary
    - Code-switched: a mix of the above within a single sentence

Used by classical features (src/features/classical_features.py), Word2Vec
training (src/features/embeddings.py), and every from-scratch neural model's
input pipeline. NOT used by the BERT benchmark, which uses its own tokenizer.

TODO:
- class CodeMixTokenizer
    - __init__(self, lowercase=True, remove_punctuation=True)
    - tokenize(self, text: str) -> list[str]
    - build_vocab(self, corpus: list[str], min_freq: int = 1) -> None
        Builds self.token_to_id / self.id_to_token from a list of documents.
    - encode(self, text: str, max_length: int) -> list[int]
        Tokenize + map to ids + pad/truncate to max_length.
    - vocab_size property
- Persisting the vocab (save/load) so training and the Streamlit app use the
  same tokenizer without rebuilding it each time.
"""


class CodeMixTokenizer:
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        raise NotImplementedError

    def tokenize(self, text: str) -> list:
        raise NotImplementedError

    def build_vocab(self, corpus: list, min_freq: int = 1) -> None:
        raise NotImplementedError

    def encode(self, text: str, max_length: int) -> list:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "CodeMixTokenizer":
        raise NotImplementedError

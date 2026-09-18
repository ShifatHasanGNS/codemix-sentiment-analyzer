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
"""

import re

from src.utils.io_utils import load_json, save_json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# A run of Bangla-script letters, a run of other-script "word" letters
# (with internal apostrophes for contractions like "don't"), a run of
# digits, or any single other character (punctuation/emoji) -- each becomes
# one token. This is script-aware rather than whitespace-only, so Bangla
# text (which doesn't reliably space-separate the way English does) still
# tokenizes into meaningful units instead of one giant blob.
#
# The "other word letters" branch uses \w's general Unicode letter
# definition (via [^\W\d_...], not a plain [A-Za-z]) so stylized Unicode
# letters survive tokenization too -- a real review in this corpus writes
# "onak kharap" as "𝒐𝒏𝒂𝒌 𝒌𝒉𝒂𝒓𝒂𝒑" using Mathematical Bold Italic glyphs,
# which are letters by Unicode's own classification. An ASCII-only pattern
# would silently drop that review to zero tokens -- an all-padding input
# that drives AdditiveAttention's softmax to 0/0 = NaN and permanently
# corrupts the model's weights via backprop (see AdditiveAttention's
# defensive masking, which handles any that still slip through at
# inference time).
_LETTER_RUN = r"[^\W\d_ঀ-৿\s]+(?:'[^\W\d_ঀ-৿\s]+)*"
_TOKEN_RE = re.compile(
    r"[ঀ-৿]+"
    rf"|{_LETTER_RUN}"
    r"|\d+(?:\.\d+)?"
    r"|[^\sA-Za-zঀ-৿\d]"
)
_WORD_RE = re.compile(rf"^([ঀ-৿]+|{_LETTER_RUN}|\d+(?:\.\d+)?)$")


class CodeMixTokenizer:
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def tokenize(self, text: str) -> list:
        tokens = _TOKEN_RE.findall(text or "")
        if self.remove_punctuation:
            tokens = [t for t in tokens if _WORD_RE.match(t)]
        if self.lowercase:
            tokens = [t.lower() if t.isascii() else t for t in tokens]
        return tokens

    def build_vocab(self, corpus: list, min_freq: int = 1) -> None:
        counts = {}
        for text in corpus:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        # Sort by (frequency desc, token asc) for a deterministic vocab
        # ordering across runs given the same corpus.
        kept = sorted(
            (token for token, count in counts.items() if count >= min_freq),
            key=lambda token: (-counts[token], token),
        )

        self.token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token in kept:
            self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, text: str, max_length: int) -> list:
        unk_id = self.token_to_id[UNK_TOKEN]
        pad_id = self.token_to_id[PAD_TOKEN]
        ids = [self.token_to_id.get(t, unk_id) for t in self.tokenize(text)]
        ids = ids[:max_length]
        ids += [pad_id] * (max_length - len(ids))
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str) -> None:
        save_json({
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "token_to_id": self.token_to_id,
        }, path)

    @classmethod
    def load(cls, path: str) -> "CodeMixTokenizer":
        state = load_json(path)
        tokenizer = cls(lowercase=state["lowercase"], remove_punctuation=state["remove_punctuation"])
        tokenizer.token_to_id = state["token_to_id"]
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.token_to_id.items()}
        return tokenizer

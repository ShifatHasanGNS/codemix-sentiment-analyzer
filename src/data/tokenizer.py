"""Script-aware tokenizer for mixed Bangla-English text. Used by classical features, Word2Vec, and from-scratch neural models (not BERT, which has its own)."""

import re

from src.utils.io_utils import load_json, save_json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# Script-aware (not whitespace-only) so Bangla text still tokenizes into units; \W-based
# letter run (not plain [A-Za-z]) so stylized Unicode letters aren't dropped to zero tokens.
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

        # Deterministic order: frequency desc, token asc.
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

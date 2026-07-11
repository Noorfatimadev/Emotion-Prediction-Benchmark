import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Tokenization

def _simple_tokenize(text: str) -> List[str]:
    """
    Minimal tokenizer for a BoW / embedding baseline.

    - Keeps letters, numbers, and apostrophes (e.g., "don't")
    - Ignores punctuation and whitespace
    """
    return re.findall(r"[A-Za-z0-9']+", text)


# Label column handling
def _get_label_column(df: pd.DataFrame) -> str:
    """
    Return the regression label column.

    Raises an error if doesnt exists.
    """
    if "normalized" in df.columns:
        return "normalized"

    if "V" in df.columns:
        return "V"

    raise ValueError(
        "No valid label column found. Expected 'normalized' or 'V'. "
        f"Available columns: {list(df.columns)}"
    )


# BoW Dataset

class BowDataset(Dataset):
    """
    Bag-of-words dataset with multiple feature variants:
      - count   : raw token counts
      - binary  : 0/1 for token presence
      - log_tf  : log(1 + count)
      - tfidf   : tf * idf   (idf must be fit on TRAIN only)

    __getitem__ returns:
      x: FloatTensor [V]  (V = bow_vocab_size)
      y: FloatTensor [1]  (regression label)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bow_vocab_size: int,
        data_col: str,
        split: str,
        lowercase: bool,
        vocab: Optional[Dict[str, int]] = None,
        bow_variant: str = "count",
        idf: Optional[np.ndarray] = None,
        label_col: Optional[str] = "normalized",
    ):
        # Keep indices clean/consistent for __getitem__
        self.df = df.reset_index(drop=True)

        # Which text field we featurize (e.g., "text", "lemma", "PoS")
        self.data_col = data_col
        self.lowercase = lowercase

        # Feature dimensions are fixed to bow_vocab_size
        self.vocab_size = bow_vocab_size
        self.bow_variant = bow_variant.lower()

        # Decide which label column to use (or detect if not provided)
        if label_col is None:
            label_col = _get_label_column(df)
        if label_col not in df.columns:
            raise ValueError(f"label_col='{label_col}' not in df columns: {list(df.columns)}")
        self.label_col = label_col

        # Vocabulary must be created on TRAIN only and reused for dev/test
        if vocab is None:
            self.vocab = self._build_vocab(self.df[self.data_col].tolist(), bow_vocab_size)
        else:
            self.vocab = vocab

        # Ensure there is always an UNK index for unseen tokens
        if "[UNK]" not in self.vocab:
            # Put UNK in the last available slot (or append if there is room)
            unk_idx = min(len(self.vocab), bow_vocab_size - 1)
            self.vocab["[UNK]"] = unk_idx
        self.unk_idx = self.vocab["[UNK]"]

        # TF-IDF requires an IDF vector:
        # - compute it on TRAIN
        # - pass it in for DEV/TEST to avoid leakage
        self.idf = None
        if self.bow_variant == "tfidf":
            if split == "train":
                self.idf = self._compute_idf(self.df[self.data_col].tolist())
            else:
                if idf is None:
                    raise ValueError("TF-IDF requires idf computed on train and passed for dev/test.")
                self.idf = idf.astype(np.float32)

    def _build_vocab(self, texts: List[str], max_size: int) -> Dict[str, int]:
        """
        Build a top-K vocabulary from the training texts.

        Implementation details:
        - counts tokens across all documents
        - keeps the most frequent tokens
        - reserves one slot for [UNK] when possible
        """
        counter = Counter()

        for t in texts:
            # Handling of NaN / non-string values
            if not isinstance(t, str):
                t = "" if pd.isna(t) else str(t)

            if self.lowercase:
                t = t.lower()

            counter.update(_simple_tokenize(t))

        # Keep max_size-1 tokens to ensure room for [UNK]
        take = max_size - 1 if max_size > 1 else 1
        most_common = [w for w, _ in counter.most_common(take)]

        vocab = {w: i for i, w in enumerate(most_common)}

        # Place [UNK] either at the end (if room) or in the last slot
        if len(vocab) < max_size:
            vocab["[UNK]"] = len(vocab)
        else:
            vocab["[UNK]"] = max_size - 1

        return vocab

    def _compute_idf(self, texts: List[str]) -> np.ndarray:
        """
        Compute smoothed IDF on TRAIN texts only.

        Smooth IDF:
          idf_j = log((1 + N) / (1 + df_j)) + 1

        where:
          N    = number of documents
          df_j = number of documents containing token j at least once
        """
        N = len(texts)
        df_counts = np.zeros(self.vocab_size, dtype=np.int64)

        for t in texts:
            if not isinstance(t, str):
                t = "" if pd.isna(t) else str(t)
            if self.lowercase:
                t = t.lower()

            # Use a set so a token counts max once per document for df
            toks = set(_simple_tokenize(t))

            for tok in toks:
                j = self.vocab.get(tok, self.unk_idx)
                if 0 <= j < self.vocab_size:
                    df_counts[j] += 1

        idf = np.log((1.0 + N) / (1.0 + df_counts)) + 1.0
        return idf.astype(np.float32)

    def get_idf(self) -> Optional[np.ndarray]:
        """Expose IDF so the trainer can pass it into dev/test datasets."""
        return self.idf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create the BoW vector.

        Steps:
        1) tokenize text
        2) accumulate raw counts into x
        3) apply variant transform (binary/log_tf/tfidf)
        """
        row = self.df.iloc[idx]

        text = row[self.data_col]
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        if self.lowercase:
            text = text.lower()

        toks = _simple_tokenize(text)

        # Dense BoW vector
        x = torch.zeros(self.vocab_size, dtype=torch.float32)

        # Raw term frequency counts
        for tok in toks:
            j = self.vocab.get(tok, self.unk_idx)
            if 0 <= j < self.vocab_size:
                x[j] += 1.0

        # Transform counts into requested feature variant
        if self.bow_variant == "binary":
            x = (x > 0).float()
        elif self.bow_variant == "log_tf":
            x = torch.log1p(x)
        elif self.bow_variant == "tfidf":
            # Multiply tf by precomputed IDF (fit on train)
            idf_t = torch.tensor(self.idf, dtype=torch.float32)
            x = x * idf_t
        elif self.bow_variant == "count":
            pass
        else:
            raise ValueError(f"Unknown bow_variant: {self.bow_variant}")

        # Regression label
        y = torch.tensor(float(row[self.label_col]), dtype=torch.float32).view(1)
        return x, y


# ========================= Embeddings Dataset =========================

class EmbeddingsDataset(Dataset):
    """
    Dataset that returns variable-length token index sequences for embedding models.

    OOV (out-of-vocabulary) handling:
      - 'unk'  : map OOV tokens to [UNK]
      - 'skip' : drop OOV tokens (and fall back to [UNK] if everything is dropped)

    __getitem__ returns:
      seq: LongTensor [T]
      y: FloatTensor [1]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        word2vec: Any,
        data_col: str,
        lowercase: bool,
        oov_strategy: str = "unk",
        label_col: Optional[str] = "normalized",
    ):
        self.df = df.reset_index(drop=True)
        self.word2vec = word2vec
        self.data_col = data_col
        self.lowercase = lowercase
        self.oov_strategy = oov_strategy.lower()

        # Decide label column (or detect if not provided)
        if label_col is None:
            label_col = _get_label_column(df)
        if label_col not in df.columns:
            raise ValueError(f"label_col='{label_col}' not in df columns: {list(df.columns)}")
        self.label_col = label_col

        # Special tokens must exist in the embedding vocabulary
        self.unk_idx = self.word2vec.get_index("[UNK]")
        self.pad_idx = self.word2vec.get_index("[PAD]")

        # Some embeddings expose a dict; others only support get_index()
        self._has_key_to_index = hasattr(self.word2vec, "key_to_index")

    def __len__(self) -> int:
        return len(self.df)

    def _in_vocab(self, tok: str) -> bool:
        """Check whether token exists in the embedding vocabulary."""
        if self._has_key_to_index:
            return tok in self.word2vec.key_to_index
        try:
            self.word2vec.get_index(tok)
            return True
        except Exception:
            return False

    def _token_to_idx(self, tok: str) -> int:
        """Convert token to embedding index, falling back to UNK when missing."""
        try:
            return self.word2vec.get_index(tok)
        except Exception:
            if self._has_key_to_index and tok in self.word2vec.key_to_index:
                return self.word2vec.key_to_index[tok]
            return self.unk_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert one text example into a sequence of token indices.
        """
        row = self.df.iloc[idx]

        text = row[self.data_col]
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        if self.lowercase:
            text = text.lower()

        toks = _simple_tokenize(text)
        ids: List[int] = []

        # Apply OOV strategy
        if self.oov_strategy == "skip":
            for t in toks:
                if self._in_vocab(t):
                    ids.append(self._token_to_idx(t))
        elif self.oov_strategy == "unk":
            ids = [self._token_to_idx(t) for t in toks]
        else:
            raise ValueError(f"Unknown oov_strategy: {self.oov_strategy}")

        # Avoid empty sequences (breaks many models/collate functions)
        if len(ids) == 0:
            ids = [self.unk_idx]

        seq = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(float(row[self.label_col]), dtype=torch.float32).view(1)
        return seq, y
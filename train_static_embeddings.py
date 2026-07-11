# Train Word2Vec embeddings from a plain-text corpus (wikitext103.txt).


import os
import re
import argparse
import logging
from typing import Iterator, List, Optional

from gensim.models import Word2Vec


# Keeps words like "don't", "you're", "john's" plus alphanumerics.
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")

# WikiText has many section markers like "= Heading =" or "== Heading ==".
WIKITEXT_HEADING_RE = re.compile(r"^\s*=+\s*[^=].*?=+\s*$")


def iter_tokenized_lines(
    path: str,
    lowercase: bool = True,
    max_lines: Optional[int] = None,
    max_tokens: Optional[int] = None,
    drop_headings: bool = True,
    drop_numeric_only: bool = False,
) -> Iterator[List[str]]:

    seen_tokens = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            if drop_headings and WIKITEXT_HEADING_RE.match(line):
                continue

            if lowercase:
                line = line.lower()

            toks = TOKEN_RE.findall(line)
            if not toks:
                continue

            if drop_numeric_only:
                toks = [t for t in toks if not t.isdigit()]
                if not toks:
                    continue

            yield toks
            seen_tokens += len(toks)

            if max_tokens is not None and seen_tokens >= max_tokens:
                break


class TokenizedCorpus:
    """
    Re-iterable corpus wrapper for gensim.

    """
    def __init__(
        self,
        path: str,
        lowercase: bool = True,
        drop_headings: bool = True,
        drop_numeric_only: bool = False,
        max_lines: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.path = path
        self.lowercase = lowercase
        self.drop_headings = drop_headings
        self.drop_numeric_only = drop_numeric_only
        self.max_lines = max_lines
        self.max_tokens = max_tokens

    def __iter__(self) -> Iterator[List[str]]:
        yield from iter_tokenized_lines(
            self.path,
            lowercase=self.lowercase,
            max_lines=self.max_lines,
            max_tokens=self.max_tokens,
            drop_headings=self.drop_headings,
            drop_numeric_only=self.drop_numeric_only,
        )


def count_tokens_quick(corpus: TokenizedCorpus) -> int:
    """One pass token count."""
    total = 0
    for toks in corpus:
        total += len(toks)
    return total


def build_argparser(default_corpus: str, default_out: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--corpus_path",
        default=default_corpus,
        help="Path to plain-text corpus (default: wikitext103.txt next to this script)",
    )
    ap.add_argument(
        "--out_path",
        default=default_out,
        help="Output embeddings path (.bin, word2vec binary format)",
    )

    # Lowercasing: default ON
    ap.add_argument("--no_lowercase", action="store_true", help="Disable lowercasing")

    # WikiText cleanup toggles
    ap.add_argument("--keep_headings", action="store_true", help="Keep heading lines like '== Section =='")
    ap.add_argument("--drop_numeric_only", action="store_true", help="Drop tokens that are only digits")

    # Debug/speed controls
    ap.add_argument("--max_lines", type=int, default=None, help="Limit lines (debug)")
    ap.add_argument("--max_tokens", type=int, default=None, help="Stop after ~N tokens (speed control)")
    ap.add_argument("--check_token_count", action="store_true", help="Count tokens before training (slower)")

    # Hyperparameters
    ap.add_argument("--dim", type=int, default=300, help="Vector dimension")
    ap.add_argument("--window", type=int, default=5, help="Context window size")
    ap.add_argument("--min_count", type=int, default=5, help="Min word frequency")
    ap.add_argument("--workers", type=int, default=8, help="CPU workers")
    ap.add_argument("--epochs", type=int, default=5, help="Training epochs")
    ap.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    ap.add_argument("--negative", type=int, default=10, help="Negative samples")
    ap.add_argument("--sample", type=float, default=1e-4, help="Subsampling threshold")

    return ap


def main():
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    log = logging.getLogger("train_word2vec")

    here = os.path.dirname(os.path.abspath(__file__))
    default_corpus = os.path.join(here, "wikitext103.txt")
    default_out_dir = os.path.join(here, "embeddings")
    os.makedirs(default_out_dir, exist_ok=True)
    default_out = os.path.join(default_out_dir, "wikitext103_w2v_300d.bin")

    ap = build_argparser(default_corpus, default_out)
    args = ap.parse_args()

    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(
            f"Corpus file not found:\n  {args.corpus_path}\n\n"
            f"Expected default location:\n  {default_corpus}\n"
            f"Or pass --corpus_path to point to the file."
        )

    lowercase = not args.no_lowercase
    drop_headings = not args.keep_headings

    corpus = TokenizedCorpus(
        args.corpus_path,
        lowercase=lowercase,
        drop_headings=drop_headings,
        drop_numeric_only=args.drop_numeric_only,
        max_lines=args.max_lines,
        max_tokens=args.max_tokens,
    )

    if args.check_token_count:
        log.info("Counting tokens...")
        total_tokens = count_tokens_quick(corpus)
        log.info(f"Token count (approx): {total_tokens:,}")

        if total_tokens < 10_000_000 and args.max_lines is None and args.max_tokens is None:
            log.warning("Corpus appears to have <10M tokens.")

        corpus = TokenizedCorpus(
            args.corpus_path,
            lowercase=lowercase,
            drop_headings=drop_headings,
            drop_numeric_only=args.drop_numeric_only,
            max_lines=args.max_lines,
            max_tokens=args.max_tokens,
        )

    # Build Word2Vec model
    model = Word2Vec(
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        negative=args.negative,
        sample=args.sample,
    )

    log.info("Building vocabulary...")
    model.build_vocab(corpus)

    log.info("Training embeddings...")
    model.train(corpus, total_examples=model.corpus_count, epochs=args.epochs)

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    log.info(f"Saving embeddings to: {args.out_path}")
    model.wv.save_word2vec_format(args.out_path, binary=True)

    log.info("Done.")


if __name__ == "__main__":
    main()
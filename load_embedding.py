#!/bin/env python3
# coding: utf-8

# To run this on Fox, load this module:
# nlpl-nlptools/01-foss-2024a-Python-3.12.3

import gensim

# Models can be found at https://vectors.nlpl.eu/explore/embeddings/models/,
# or https://vectors.nlpl.eu/repository/ (don't forget to unzip the model),
# or in the /fp/projects01/ec403/models/static/  directory on Fox
# (for example, /fp/projects01/ec403/models/static/223/)


#!/bin/env python3
# coding: utf-8

import gensim


def load_embedding(modelfile):
    # Facebook FastText model
    if modelfile.endswith("parameters.bin"):
        emb_model = gensim.models.fasttext.load_facebook_vectors(modelfile)

    # Binary word2vec format:
    elif modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )

    # Text word2vec / vec / glove text format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        try:
            # Try standard word2vec text (expects header line)
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                modelfile, binary=False, unicode_errors="replace"
            )
        except ValueError:
            # Fallback for GloVe text (no header line)
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                modelfile, binary=False, unicode_errors="replace", no_header=True
            )

    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)

    return emb_model
"""Generate a DDR lexicon.

Outputs to shifterator/lexicons/DDR

The vocabulary will be limited to whatever words are in the word2vec model.
And scores are derived by similarity to the mean DDR vector.
"""
import os
import time
import argparse

import numpy as np
import pandas as pd

from gensim import downloader

# all currently available in gensim
MODEL_FULLNAMES = {
    "fasttext300"   : "fasttext-wiki-news-subwords-300",
    "conceptnet300" : "conceptnet-numberbatch-17-06-300",
    "ruscorpora"    : "word2vec-ruscorpora-300",
    "gnews300"      : "word2vec-google-news-300",
    "wiki50"        : "glove-wiki-gigaword-50",
    "wiki100"       : "glove-wiki-gigaword-100",
    "wiki200"       : "glove-wiki-gigaword-200",
    "wiki300"       : "glove-wiki-gigaword-300",
    "twitter25"     : "glove-twitter-25",
    "twitter50"     : "glove-twitter-50",
    "twitter100"    : "glove-twitter-100",
    "twitter200"    : "glove-twitter-200",
}

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True)
parser.add_argument("-m", "--model", type=str, required=True, choices=MODEL_FULLNAMES.keys())
parser.add_argument("-w", "--words", nargs="+", type=str, required=True)
args = parser.parse_args()
# parser.add_argument('-n', '--names-list', nargs='+', default=[])


DICTIONARY_NAME = args.name
MODEL = args.model
DICTIONARY_WORDS = args.words


EXPORT_DIR = "lexicons/DDR"

# create the DDR folder is not present
os.makedirs(EXPORT_DIR, exist_ok=True)

export_scores_fname = os.path.join(EXPORT_DIR, f"DDR_{DICTIONARY_NAME}-{MODEL}.tsv")
export_dict_fname = export_scores_fname.replace(".tsv", ".txt")

####### load pretrained model
model_fullname = MODEL_FULLNAMES[MODEL]
print(f"Loading model {model_fullname}...")
t0 = time.time()
word2vec = downloader.load(model_fullname)
t1 = time.time()
print("Model loaded in {:.1f} minutes.".format((t1-t0)/60))
word2vec.init_sims(replace=True)
# model.delete_temporary_training_data(replace_word_vectors_with_normalized=True)

# get main DDR vector representation by averaging all relevant vectors
ddr_vector = np.mean([ word2vec.get_vector(w) for w in DICTIONARY_WORDS ], axis=0)

vocabulary = [ tok for tok in word2vec.vocab.keys() if tok.isalpha() ]

word_vectors = np.vstack([ word2vec.get_vector(w) for w in vocabulary ])

word_similarities = word2vec.cosine_similarities(ddr_vector, word_vectors)

df = pd.DataFrame({"word":vocabulary, "score":word_similarities})
df = df.sort_values("score", ascending=False)

df.to_csv(export_scores_fname, sep="\t", index=False, header=None)

# write a txt file denoting the words in the dict
with open(export_dict_fname, "w") as f:
    f.write("\n".join(DICTIONARY_WORDS))

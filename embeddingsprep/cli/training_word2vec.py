"""
                            Training Word2Vec

Usage:
    training_word2vec.py <file_dir> <writing_dir>
 
Options:
    -h --help
    --version
    --file_dir                File directories
    --writing_dir             Writing directory
"""
import os
import gensim
from glob import glob
from docopt import docopt

from models.utils import read_files, open_vocab

if __name__ == "__main__":
    args = docopt(__doc__, version="0.1")
    print(args)
    filenames = glob(os.path.join(args["<file_dir>"], "*.txt"))
    writing_dir = args["<writing_dir>"]
    modelname = glob(os.path.join(writing_dir, "*.model"))
    if len(modelname) == 1:
        model = gensim.models.Word2Vec.load(modelname[0])
    else:
        sents = read_files(filenames)
        model = gensim.models.Word2Vec(
            sents, size=200, window=10, min_count=2, workers=16
        )
        model.train(sents, total_examples=len(sents), epochs=25)
        model.save(os.path.join(writing_dir, "word2vec.model"))

    try:
        vocab_path = glob(os.path.join(args["<file_dir>"], "*.json"))[0]
        vocab = open_vocab(vocab_path)
    except:
        vocab_keys = list(model.wv.vocab.keys())
        vocab = {vocab_keys[i]: i for i in range(len(vocab_keys))}

    inv_vocab = {v: k for k, v in vocab.items()}
    METADATA_PATH = os.path.join(writing_dir, "metadata.tsv")
    VECTOR_PATHS = os.path.join(writing_dir, "embeddings.tsv")
    with open(METADATA_PATH, "w", encoding="utf-8") as metadata:
        with open(VECTOR_PATHS, "w", encoding="utf-8") as vectors:
            metadata.write("WORD\tINDEX\n")
            for i in range(len(vocab)):
                try:
                    vector = model.wv[inv_vocab[i]]
                    metadata.write(str(inv_vocab[i]) + "\t" + str(i) + "\n")
                    n = len(vector)
                    for j in range(n):
                        if j == (n - 1):
                            vectors.write(str(vector[j]) + "\n")
                        else:
                            vectors.write(str(vector[j]) + "\t")
                except:
                    print(
                        "{0} not in vocabulary. Passing. \n".format(
                            inv_vocab[i]
                        )
                    )
                    pass

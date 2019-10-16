"""
                                Multiprocessed Preprocessing

This preprocesses text with multiprocessing.

Usage:
  preprocessing.py <file_dir> <writing_dir> [--batch_size=<bs>] 
  [--phrases_threshold=<pt>] [--freq_threshold=<ft>] [--vocabulary_size=<vs>] 
  [--wiki=<w>] [--simple_file=<sf>]
 
Options:
  -h --help
  --version
  --file_dir                File directories, where files to be preprocessed are stored
  --writing_dir             Writing directory, where the preprocessed files will be stored
  --batch_size=<bs>         The batch size for batch preprocessing. [default: 400].
  --phrases_threshold=<pt>  Phrases threshold for bigrams gathering [default: 0.001].
  --freq_threshold=<ft>     Frequency threshold for discarding a word from vocabulary dictionnary. [default: 0.01].
  --vocabulary_size=<vs>    The maximum vocabulary size for a vocabulary dictionnary [default: 100000]
"""

from glob import glob
import os
import sys
from docopt import docopt

sys.path.append("../preprocessing")
sys.path.append('../utils')
from utils.readers import readArgs
from utils.preproc_multiproc import Preprocessing


def preprocess(
    file_dir,
    writing_dir,
    batch_size,
    phrases_threshold,
    freq_threshold,
    vocabulary_size,
    wiki,
    simple_file,
):
    path = os.path.join(file_dir, "*")
    filenames = glob(path)  # Filenames is a list of directories
    # Path manipulation : creating the folders to write files
    path_cleaned = os.path.join(writing_dir, "cleaned")
    path_vocabulary = os.path.join(writing_dir, "vocabulary")

    if os.path.exists(path_cleaned):
        os.removedirs(path_cleaned)

    if os.path.exists(path_vocabulary):
        os.removedirs(path_vocabulary)

    os.makedirs(path_cleaned)
    os.makedirs(path_vocabulary)

    truc = Preprocessing(
        phrases_threshold=phrases_threshold,
        n_iter_phrases=1,
        phrases_delta=1,
        freq_threshold=freq_threshold,
        filenames=filenames,
        writing_dir=path_cleaned,
        vocab_dir=path_vocabulary,
        disable_subsampling=True,
        vocabulary_size=vocabulary_size
    )

    truc.fit_transform()
    return None


if __name__ == "__main__":
    args = docopt(__doc__, version="0.1")
    params = readArgs(args)
    print(args)
    preprocess(
        params["file_dir"],
        params["writing_dir"],
        params["batch_size"],
        params["phrases_threshold"],
        params["freq_threshold"],
        params["vocabulary_size"]
    )

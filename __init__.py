"""
# Embeddings


Embedding generation with text preprocessing.


## Preprocessor

For Word2Vec, we want a soft yet important preprocessing. We want to denoise the text while keeping as much variety and information as possible.

Preprocesses the text/set of text in the following way :

1. Detects and replaces numbers/float by a generic token 'FLOAT', 'INT'

2. Add spaces in between punctuation so that tokenisation avoids adding 'word.' to the vocabulary instead of 'word', '.'

3. Lowers words

4. Recursive word phrases detection : with a simple probabilistic rule, gathers the tokens 'new', york' to a single token 'new_york'.

5. Frequency Subsampling : discards unfrequent words with a probability depending on their frequency.


 Outputs a vocabulary file and the modified files.

Usage example :


```python
from preprocessing.preprocessor import PreprocessorConfig, Preprocessor
config = PreprocessorConfig('/tmp/logdir')
config.set_config(writing_dir='/tmp/outputs')
config.save_config()


prep = Preprocessor('/tmp/logdir')
prep.fit('~/mydata/')
prep.get_summary()
prep.save_word_phrases()
prep.transform('~/mydata')
```


##  Word2Vec

For the Word2Vec, we just wrote a simple cli wrapper that takes the
preprocessed files as an input, trains a Word2Vec model with gensim and writes the vocab, embeddings .tsv files that can be visualized with tensorflow projector (http://projector.tensorflow.org/)


Usage example:

```bash
python training_word2vec.py file_dir writing_dir
```

TODO :
- [ ] Clean code for CLI wrapper
- [ ] Also write a python Word2Vec model class so that user doesn't have to switch from python to cli
- [ ] Also write a cli wrapper for preprocessing
"""
name = "embeddings"

# Embeddings


This package is designed to provide easy-to-use python class and cli
interfaces to:

- clean corpuses in an efficient way in terms of computation time

- generate word2vec embeddings (based on gensim) and directly write them to a format that is compatible with [Tensorflow Projector](http://projector.tensorflow.org/)

Thus, with two classes, or two commands, anyone should be able clean a corpus and generate embeddings that can be uploaded and visualized with Tensorflow Projector.

## Getting started

### Requirements

This packages requires ```gensim```, ```nltk```, and ```docopt``` to run. If
pip doesn't install this dependencies automatically, you can install it by
running :
```bash
pip install nltk docopt gensim
```

### Installation

To install this package, simply run :
```bash
pip install embeddings-prep
```

Further versions might include conda builds, but it's currently not the case.


## Main features

### Preprocessing

For Word2Vec, we want a soft yet important preprocessing. We want to denoise the text while keeping as much variety and information as possible. A detailed version of what is done during the preprocessing is available [here](./preprocessing/index.html)


#### Usage example :

Creating and saving a loadable configuration:
```python
from embeddings.preprocessing.preprocessor import PreprocessorConfig, Preprocessor
config = PreprocessorConfig('/tmp/logdir')
config.set_config(writing_dir='/tmp/outputs')
config.save_config()
```

```python
prep = Preprocessor('/tmp/logdir')  # Loads the config object in /tmp/logdir if it exists
prep.fit('~/mydata/')  # Fits the unigram & bigrams occurences
prep.filter()  # Filters with all the config parameters
prep.transform('~/mydata')  # Transforms the texts with the filtered vocab. 
```


### Word2Vec

For the Word2Vec, we just wrote a simple wrapper that takes the
preprocessed files as an input, trains a Word2Vec model with gensim and writes the vocab, embeddings .tsv files that can be visualized with tensorflow projector (http://projector.tensorflow.org/)

#### Usage example:


```python
from models.word2vec import Word2Vec
model = Word2Vec(emb_size=300, window=5, epochs=3)
model.train('./my-preprocessed-data/')
model.save('./my-output-dir')
```

## Contributing

Any github issue, contribution or suggestion is welcomed! You can open issues on the [github repository](https://github.com/sally14/embeddings).

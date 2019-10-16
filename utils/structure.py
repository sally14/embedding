"""
--------------------------------------------------------------------------------
                                Structure Utils
--------------------------------------------------------------------------------

Functions changing the text data structure
"""

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import ngrams
from collections import Counter
from itertools import chain
from json import loads

null = "NULL"


def get_sentences(text, sentencizer=None):
    """
    Args:
       text:
            a string
       sentencizer:
            a function to parse sentences.
            Default: nltk sent_tokenize.
    Returns :
        a list containing sentences, which are lists of strings.

    """
    if sentencizer is None:
        sent_list = sent_tokenize(text)
        word_list = [word_tokenize(i) for i in sent_list]
        return word_list
    else:
        return sentencizer(text)


def get_words(text):
    """
    Args :
        text : a string
    Returns :
        a list containing words, with nltk word tokenize.

    """
    return word_tokenize(text)


def get_text(sentences, lower_cased=False):
    """
    Args :
        sentences :
            a list containing sentences, which are lists of strings.
    Returns :
        the string of the whole text.

    """
    texts = [" ".join(i) for i in sentences]
    text = " ".join(texts)
    if lower_cased:
        return text.lower()
    else:
        return text


def get_vocab(words):
    """
    Args :
        sents : list of the sentences in the text
    Returns :
        dict : {word : frequency}
    """
    dico = dict(Counter(words))  # Building vocab dict
    return dico


def get_ngrams(sentences, n):
    """
    Args :
        sentences : list of sentences (list of word-strings)
        n : ngrams size

    Returns :
        get_ngrams(sentences,n)[i]  = list of i'th sentence ngrams
    """
    ngram = []
    for i in sentences:  # Iteration over sentences
        # i = [null for i in range(n-1)] + i + [null for i in range(n-1)]
        gram = ngrams(i, n)
        lgram = [j for j in gram]
        ngram.append(lgram)
    return ngram


def ngrams2sents(grams):
    """
    Args :
        ngrams :
            a list of ngrams as n-uplets, listed by sentences.
            (see : get_ngrams)
    Returns :
        sentences :
            a list of sentences (see : get_sentences)
    """
    n = len(grams[0][0])
    nb_sents = len(grams)
    ls_sents = []
    for i in range(nb_sents):  # Iterating on sentences
        ls_words = [grams[i][0][j] for j in range(n)]
        len_sent = len(grams[i])
        for j in range(1, len_sent):
            ls_words.append(grams[i][j][-1])
            ls_sents.append(ls_words)
    return ls_sents


def sent2words(sent):
    """
    Args :
        sent :
            list of sentences (list of word-strings)
    Returns :
        words :
            a list of the words
    """
    return list(chain.from_iterable(sent))


def melt_vocab_dic(vocab1, vocab2):
    for word in vocab2:
        if word in vocab1:
            vocab1[word] = vocab1[word] + vocab2[word]
        else:
            vocab1[word] = vocab2[word]
    return vocab1


def file2sent(filename, type="txt", del_refs=True):
    """
    Args :
        file : str
            filename
        type: str
            'wiki': json files coming from wikipedia scrapping.
                    One file contains one article per line in field 'text'
            'txt': txt file name, with 1 sentence per line in the .txt file.
    Returns :
        sent :
            list of sentences (list of word-strings)
    """
    with open(filename, "r", encoding="utf-8") as file:
        sent2 = []
        if type == "txt":
            must_pass = False
            for sent in file:
                if "==== Front" in sent:
                    must_pass = True
                elif "==== Body" in sent:
                    must_pass = False
                elif must_pass:
                    pass
                elif "==== Refs" in sent and del_refs:
                    break
                elif len(sent.split()) > 0:
                    sent2.append(sent.split())
            return sent2
        elif type == "wiki":
            for sent in file:
                # Recover text article from wikipedia data
                # Split twice with \n to get rid off wiki title
                text = loads(sent)["text"].split("\n", 2)[-1]
                # Extend sent2 with sentences
                sent2.extend(
                    [t.split() for t in text.split(".") if len(t) > 0]
                )
            return sent2

"""
--------------------------------------------------------------------------------
                        Multiprocessed Preprocessing
--------------------------------------------------------------------------------


Cleaning/Preprocessing functions
"""

from utils.structure import get_vocab
from utils.structure import melt_vocab_dic
from utils.structure import get_ngrams
from utils.structure import sent2words
from utils.structure import file2sent as file2sent_txt
from collections import OrderedDict
from hashlib import sha1
from random import seed
from random import uniform
from numpy import sqrt

# from glob import glob
from string import punctuation
from multiprocessing import Pool
from multiprocessing import cpu_count
import json
import gc
import os


cpu = cpu_count()


class PreprocessorConfig(object):
    

class Preprocessing(object):
    def __init__(
        self,
        n_iter_phrases=1,
        freq_threshold=1e-5,
        phrases_delta=0,
        phrases_threshold=1e-3,
        # data_dir='',
        filenames="",
        writing_dir="",
        vocabulary_size=None,
        del_punctuation=True,
        disable_subsampling=True,
        simple_file=False,
        del_refs=True,
        vocab_dir=None,
        nb_batch=None,
        wiki=True,
    ):
        self.n_iter_phrases_ = n_iter_phrases  # Nb. of wordphrases iterations
        self.parsing_char_ = sha1(b"Posos").hexdigest()  # Hash parsing char
        self.phrases_delta_ = phrases_delta  # Delta in Mik. wordphrases (WP)
        self.phrases_threshold_ = phrases_threshold  # Threshold in Mik. WP
        self.freq_threshold_ = freq_threshold  # Freq. subsampling threshold
        self.unigram_dic_ = {}  # Stores unigrams & their occurences
        self.unigram_freq_ = {}  # Stores unigrams & their frequencies
        self.bigram_dic_ = {}  # Stores bigrams & their occurences
        self.vocabulary_ = {"_-_OOV_-_": 0}
        self.vocabulary_size_ = vocabulary_size
        # Stores final vocab. (after WP : uni&bi-grams)
        self.phrasewords_ = {}  # Stores word phrases
        self.filenames = filenames
        self.wiki = wiki
        if self.wiki:
            self.file2sent = file2sent_wiki
        else:
            self.file2sent = file2sent_txt
        # self.data_dir_ = data_dir  # Data directory
        # if not os.path.exists(writing_dir):
        #     os.makedirs(writing_dir)
        self.writing_dir_ = writing_dir  # Directory for writing prep. data
        self.del_punctuation_ = del_punctuation  # If True, delete punctuation
        self.disable_subsampling_ = disable_subsampling  # If True, no subsamp.
        self.simple_file_ = simple_file  # True if the data_dir corresponds to
        # a single file (and not a directory)
        self.del_refs = del_refs
        if vocab_dir is not None:
            self.vocab_dir = vocab_dir
        else:
            self.vocab_dir = self.writing_dir_
        if nb_batch is not None:
            self.nb_batch = str(nb_batch)
        else:
            self.nb_batch = ""

    def delete_punctuation(self, sentences):
        """
        Args :
            sentences :
                list of sentences (list of word (strings))
        Returns :
            sentences :
                list of sentences, without any punctuation
        """
        pun = punctuation
        translator = str.maketrans(
            " ", " ", pun.replace("_", "").replace("-", "")
        )
        # Only delete punctuation if del_punctuation is True
        if self.del_punctuation_:
            new_sent = []
            for sent in sentences:
                new_word = []
                for word in sent:
                    # str.translate() function is quicker than everything else
                    new_word.append(word.translate(translator))
                new_sent.append(new_word)
        return new_sent

    def get_unigram_voc(self, sentences):
        """
        Builds a dictionnary, batch per batch, of unique unigrams & their
        occurences.
        Args :
            sentences :
                list of sentences (list of word (strings))
        Returns :
            None
        """
        words = sent2words(sentences)
        vocabulary = get_vocab(words)  # Coded with Counter (collections)
        del words
        # Use melt_vocab_dic in order to be able to update occurences.
        return vocabulary

    def get_bigram_voc(self, sentences):
        """
        Builds a dictionnary, batch per batch, of unique bigrams & their
        occurences.
        Args :
            sentences :
                list of sentences (list of word (strings))
        Returns :
            None
        """
        bigrams = get_ngrams(sentences, 2)  # Generating bigrams
        for i in range(len(bigrams)):
            for j in range(len(bigrams[i])):
                # Gathering bigrams with a hash string 'parsing_char'
                # Hash strings prevents catching a "_" that could be in corpus
                bigrams[i][j] = self.parsing_char_.join(bigrams[i][j])
        words = sent2words(bigrams)
        vocabulary = get_vocab(words)
        del words
        del bigrams
        # Format vocabulary to be able to melt it with
        # {bigram : (occurences, score)} when iterating word phrases.
        # melt_vocab_dic adds occurences, so at step n, 0 will add nothing to
        # the score that was computed at step n-1.
        for key in vocabulary:
            vocabulary[key] = (vocabulary[key], 0)
        return vocabulary

    def build_score(self):
        """
        Add bigram score to the 'bigram_dic_' dictionnary.
        bigram_dic_ = {bigram : occurences} becomes:
        bigram_dic_ = {bigram : (occurences, score)}
        Args :
            None
        Returns :
            None
        """
        for bigrams in self.bigram_dic_:
            i, j = bigrams.split(self.parsing_char_)
            score = (self.bigram_dic_[bigrams][0] - self.phrases_delta_) / (
                self.unigram_dic_[i] * self.unigram_dic_[j]
            )
            self.bigram_dic_[bigrams] = (self.bigram_dic_[bigrams][0], score)

    def build_vocab(self):
        """
        Create a dictionnary 'vocabulary_' which contains unigrams and word
        phrases, with their occurences.
        Args :
            None
        Returns :
            None
        """
        copy_dict = self.unigram_dic_.copy()
        for word in self.bigram_dic_:
            # First feed the vocabulary with bigrams :
            if word in self.phrasewords_:
                try:
                    i, j = (word.replace(self.parsing_char_, " ", 1)).split()
                    # delete unigrams if unigrams only appear in a given bigram
                    if self.unigram_dic_[i] == self.phrasewords_[word]:
                        try:
                            # Delete element from copy_dict and not
                            # unigram_dic_
                            del copy_dict[i]
                        except:
                            pass
                    if self.unigram_dic_[j] == self.phrasewords_[word]:
                        try:
                            del copy_dict[j]
                        except:
                            pass
                    self.vocabulary_[
                        word.replace(self.parsing_char_, "_")
                    ] = self.phrasewords_[word]
                except:
                    pass
        self.vocabulary_ = melt_vocab_dic(copy_dict, self.vocabulary_)

    def phrasewords(self):
        """
        Create a dictionnary 'phrasewords_' which contains word
        phrases, with their occurences.
        Args :
            None
        Returns :
            None
        """
        for bigrams in self.bigram_dic_:
            if self.bigram_dic_[bigrams][1] > self.phrases_threshold_:
                self.phrasewords_[bigrams] = self.bigram_dic_[bigrams][0]

    def wordcount2freq(self):
        """
        Create the 'vocab_freq_' dictionnary : goes from a vocabulary_
        dictionnary with occurences to a dictionnary of the vocabulary with
        frequencies. Useful for frenquency subsampling.
        Args :
            None
        Returns :
            None
        """
        count = 0
        dico = self.vocabulary_
        dico2 = {}
        for i in dico:
            count = count + dico[i]
        for i in dico:
            newkey = i.replace(self.parsing_char_, "_", 1)
            dico2[newkey] = dico[i] / count
        self.vocab_freq_ = dico2

    def fit(self, filenames):
        """
        Batches are the different files in 'filenames'. Fit method sees all the
        batches in filenames, to feed unigram_dic_ and bigram_dic_ with word
        occurences. Then, it computes score, frequencies, etc.
        Args :
            filenames : the list on file names (string)
        Returns :
            None
        """
        for file in filenames:  # = For each batch
            sentences = self.delete_punctuation(self.file2sent(file))
            # delete_punctuation only delete punctuation if the option is
            # activated.
            self.build_unigram_voc(sentences)
            self.build_bigram_voc(sentences)
        self.build_score()
        self.phrasewords()
        self.build_vocab()
        self.wordcount2freq()

    def fit_map(self, filebatch):
        """
        Batches are the different files in 'filenames'. Fit method sees all the
        batches in filenames, to feed unigram_dic_ and bigram_dic_ with word
        occurences. Then, it computes score, frequencies, etc.
        Args :
            filenames : the list on file names (string)
        Returns :
            None
        """
        unig = {}
        big = {}
        for file in filebatch:
            sentences = self.delete_punctuation(self.file2sent(file))
            # delete_punctuation only delete punctuation if the option is
            # activated.
            unig = melt_vocab_dic(self.get_unigram_voc(sentences), unig)
            big = melt_vocab_dic(self.get_bigram_voc(sentences), big)
            del sentences
        return [unig, big]

    def subsample_freq_dic(self):
        """
        Vocab dictionnary frequency subsampling.
        Args :
            sentences : a batch of sentences as a list of words lists.
        Returns :
            sentences : the batch of subsampled sentences
        """
        t = self.freq_threshold_
        vocab = self.vocab_freq_
        for word in self.vocab_freq_.keys():
            try:  # In some very rare cases, doesn't work
                # Computing discarding word probability (Mik. 2013)
                freq = vocab[word]
                prob = 1 - sqrt(t / freq)
                # Simulating a uniform [0,1]
                # First initiate a random seed
                seed("POSOS")  # random.seed() function hashes strings
                # Simulate a binomial B(prob)
                x = uniform(0, 1)
                if x < prob:
                    del self.vocabulary_[word]
            except:
                pass
        return None

    def wordphrases(self, sentences):
        """
        Batch-per-batch word phrases gathering (in a single token, gathered
        with _ ).
        Args :
            sentences : a batch of sentences as a list of words lists.
        Returns :
            sentences : the batch of sentences, with WP gathered
        """
        count = 0
        new_sent = []  # new_sent will store the modified sentences
        for i in sentences:  # Iterating on sentences
            new_sent_sent = []
            # new_sent_sent will store the words of modified sentences
            # First handling the case where the sentence is just one word
            # cannot generate any bigram.
            if len(i) == 1:
                new_sent_sent = i
                new_sent.append(new_sent_sent)
                del new_sent_sent
            # Then regular cases :
            else:
                j = 0
                while j < (len(i) - 1):  # = for each word in the sentence
                    big = (i[j], i[j + 1])  # getting the (j-th, j+1-th)words
                    # writing the corresponding bigram :
                    bigrams = self.parsing_char_.join(big)
                    # If the bigram is enough frequent to be gathered :
                    if bigrams in self.phrasewords_:
                        # Then add the bigram as a new word in 'new_sent_sent'
                        new_sent_sent.append("_".join(big))
                        count = count + 1  # Count the number of gathered
                        # bigrams
                        # Directly go to the j+2-th word in order to avoid
                        # repeating the j+1-th word
                        j = j + 2
                    # If the bigram is not frequent enough :
                    else:
                        if j == (len(i) - 2):
                            new_sent_sent.append(i[j])
                            new_sent_sent.append(i[j + 1])
                            j = j + 2
                        # Add j-th word
                        else:
                            new_sent_sent.append(i[j])
                            # Go to j+1-th word
                            j = j + 1
                    del big
                    del bigrams
                # Finally add the sentence to the new sentences list 'new_sent'
                new_sent.append(new_sent_sent)
                del new_sent_sent
        return new_sent

    def transform(self, filebatch, n):
        for file in filebatch:
            if n == 0:
                new_file = os.path.join(
                    self.writing_dir_,
                    file.split("/")[-1].split(".")[0] + "_cleaned" + ".txt",
                )
            else:
                new_file = file
            sentence = self.delete_punctuation(self.file2sent(file))
            sentence = self.wordphrases(sentence)
            # if n == self.n_iter_phrases_-1:
            #     if not(self.disable_subsampling_):
            #         sentence = self.subsample_freq(sentence)
            text = open(new_file, "w", encoding="utf-8")
            for i in range(len(sentence)):
                text.write(" ".join(sentence[i]) + "\n")
            text.close()
            del sentence

    def get_summary(self):
        with open(
            os.path.join(self.vocab_dir, "Summary.txt"), "w", encoding="utf-8"
        ) as text:
            text.write("Parameters: \n-------------------- \n")
            text.write(
                "n_iter_phrases = "
                + str(self.n_iter_phrases_)
                + "\n"
                + "freq_threshold = "
                + str(self.freq_threshold_)
                + "\n"
                + "phrases_delta = "
                + str(self.phrases_delta_)
                + "\n"
                + "phrases_threshold = "
                + str(self.phrases_threshold_)
                + "\n"
                # + 'data_dir = ' + str(self.data_dir_) + '\n'
                + "writing_dir = "
                + str(self.writing_dir_)
                + "\n"
                + "del_punctuation = "
                + str(self.del_punctuation_)
                + "\n"
                + "disable_subsampling = "
                + str(self.disable_subsampling_)
                + "\n \n"
            )
            text.write("Attributes: \n-------------------- \n")
            text.write(
                "len(unigram_dic_) : "
                + str(len(self.unigram_dic_))
                + "\n"
                + "len(bigram_dic_) : "
                + str(len(self.bigram_dic_))
                + "\n"
                + "len(phrasewords_) : "
                + str(len(self.phrasewords_))
                + "\n"
                + "len(vocabulary_) : "
                + str(len(self.vocabulary_))
                + "\n \n"
            )
            text.write("Bigram Dic extract :\n-------------------\n")
            dico = self.bigram_dic_
            head = dict(
                [
                    (key.replace(self.parsing_char_, "_"), dico[key])
                    for key in sorted(dico.keys())[
                        len(dico) // 2 : len(dico) // 2 + 20
                    ]
                ]
            )
            text.write(str(head))
            text.write("\n\nPhrasewords Dic extract :\n-------------------\n ")
            dico = self.phrasewords_
            head = dict(
                [
                    (key.replace(self.parsing_char_, "_"), dico[key])
                    for key in sorted(dico.keys())[
                        len(dico) // 2 : len(dico) // 2 + 20
                    ]
                ]
            )
            text.write(str(head))

    def fit_transform(
        self, nb_proc=cpu_count(), given_filebatch=None, transform=True
    ):
        """
        Global method that first iterates word phrases detection and gathering,
        then subsamples. For each iteration of WP detection, the method firstly
        fits scores & frequencies, batch per batch, and then transforms data,
        batch per batch too. It takes the data from data_dir and writes the
        cleaned data in writing_dir.
        Args :
            None
        Returns :
            None
        """
        if given_filebatch is None:
            # Get file names from the data directory :
            filenames = self.filenames
        else:
            filenames = given_filebatch
        # Create data batches to feed function maping
        print("Getting filenames...")
        batches_size = max(1, len(filenames) // nb_proc)
        batches = []
        for i in range(nb_proc):
            batch = []
            for j in range(batches_size):
                try:
                    batch.append(filenames.pop())
                except:
                    pass
            batches.append(batch)
        for n in range(self.n_iter_phrases_):
            print(
                "Entering {0}-th iteration of word phrase ".format(n + 1)
                + "recognition...\n"
            )
            print("Entering fitting phase...")
            pool = Pool(processes=nb_proc)
            results = pool.map(self.fit_map, batches)
            pool.close()
            pool.terminate()
            pool.join()

            print("Melting unigram and bigrams dictionnaries...")
            for j in range(len(results)):
                self.unigram_dic_ = melt_vocab_dic(
                    self.unigram_dic_, results[j][0]
                )
                self.bigram_dic_ = melt_vocab_dic(
                    self.bigram_dic_, results[j][1]
                )
                results[j] = 0  # Clears memory
            del results
            gc.collect()
            print("Finishing fitting...")
            self.build_score()
            self.phrasewords()
            self.build_vocab()
            self.wordcount2freq()
            self.subsample_freq_dic()
            self.save()

            if transform:
                args = [(batches[i], n) for i in range(len(batches))]
                print("Entering transform phase...")
                pool = Pool(processes=nb_proc)
                pool.starmap(self.transform, args)
                pool.close()
                pool.terminate()
                pool.join()
        print("Editing Summary...")
        self.get_summary()
        gc.collect()
        del self.vocabulary_
        gc.collect()

    def save(self):
        """
        Saves downsampled vocab, by frequency, with eventual size cut
        """
        with open(
            os.path.join(
                self.writing_dir_,
                "saved_preprocessing" + self.nb_batch + ".json",
            ),
            "w",
            encoding="utf-8",
        ) as saving:
            # Delete "" if in vocabulary :
            if "" in self.vocabulary_:
                del self.vocabulary_[""]
            # Order vocab by frequency:
            ordered = OrderedDict(
                sorted(
                    self.vocabulary_.items(), key=lambda x: x[1], reverse=True
                )
            )
            self.vocabulary = {}  # Clear old copy for memory management
            # Cut vocab:
            cut_vocab = {}
            if self.vocabulary_size_ is not None:
                if "_-_OOV_-_" not in ordered:
                    self.vocabulary_size_ = self.vocabulary_size_ - 1
                i = 0
                for word in ordered:
                    if i > self.vocabulary_size_:
                        break
                    cut_vocab[word] = ordered[word]
                    i = i + 1
            print("Total vocabulary size is {0}".format(len(cut_vocab)))
            with open(
                os.path.join(self.writing_dir_, "len_vocab.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(str(len(cut_vocab)))
            ordered = {}  # Clear old copy for memory management
            # Add OOV token (must be added AFTER sorting & cutting by frequency!)
            if "_-_OOV_-_" not in cut_vocab:
                cut_vocab["_-_OOV_-_"] = len(cut_vocab)
            cut_vocab = dict(
                zip(list(cut_vocab.keys()), [i for i in range(len(cut_vocab))])
            )
            json.dump(cut_vocab, saving)

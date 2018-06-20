import os
import numpy as np
import tensorflow as tf

class Embedder:
    """Gets word embeddings for a vocabulary and saves it to a lookup table.

    The most important methods are add_embeddings_from_txt and add_embeddings_from_list
    """

    def __init__(self, embed_module, vocabulary_embeddings_path, new_vocabulary_txt=None):
        self.embed_module = embed_module
        self.vocabulary_embeddings_path = vocabulary_embeddings_path
        self.vocabulary_embeddings = self.load_embeddings()
        self.vocabulary = self.vocabulary_embeddings.keys()
        self.new_vocabulary_txt = new_vocabulary_txt


    def load_embeddings(self):
        if os.path.exists(self.vocabulary_embeddings_path):
            with open(self.vocabulary_embeddings_path, 'r') as f:
                return json.load(f)
        else: return {}

    def save_embeddings(self):
        with open(self.vocabulary_embeddings_path, 'w') as f:
            return json.dump(self.vocabulary_embeddings, f)


    def read_vocab_txt(self):
        if os.path.exists(self.new_vocabulary_txt):
            with open(self.new_vocabulary_txt, 'r') as f:
                return f.read().splitlines()
        else: return []

    def write_vocab_txt(self, vocab_list):
        with open(self.new_vocabulary_txt, 'w') as f:
            for line in vocab_list:
                f.write('{}\n'.format(line))

    def _update_embeddings(self, new_words):
        """Gets embeddings for new_words and updates vocab_embeddings dict"""
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            print(len(new_words))
            embeddings = sess.run([self.embed_module(new_words)])[0]
            print(len(embeddings))
            print(type(embeddings))
            print(embeddings.shape)
        #new_embeddings = np.array(embeddings).tolist()
        new_embeddings_dict = dict(zip(new_words, embeddings))
        self.vocabulary_embeddings.update(new_embeddings_dict)
        self.vocabulary = self.vocabulary_embeddings.keys()

    def add_embeddings_from_txt(self, save=False):
        """Adds to vocab_embeddings dict from vocabulary stored in a txt file.

        Args:
            save: boolean to save_embeddings, defaults to False
        """
        if self.new_vocabulary_txt is None:
            raise Exception('No path to new_vocab_txt found. Please define self.new_vocabulary_txt and try again.')
        new_words = self.read_vocab_txt()
        self._update_embeddings(new_words)
        if save:
            self.save_embeddings()


    def add_embeddings_from_list(new_words, add_to_txt=False, save=False):
        """Adds to vocab_embeddings dict from supplied vocabulary list.

        Args:
            add_to_txt: boolean to add supplied words to the vocab txt saved at new_vocabulary_txt
            save: boolean to save_embeddings, defaults to False
        """

        if add_to_txt == True and self.new_vocabulary_txt is None:
            raise Exception('No path to new_vocab_txt found. Please define self.new_vocabulary_txt if you would like to save new words to txt, or set update_txt=False')
        self._update_embeddings(new_words, save=save)
        if add_to_txt:
            old_words = self.read_vocab_txt() # doesn't update with self.vocabulary! only explicitly adds intended words
            self.write_vocab_txt(sorted(old_words + new_words))

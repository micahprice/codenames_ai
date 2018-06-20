import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from itertools import permutations

class OperativeAI:

    def __init__(self, embed_module, game_words=None, metric='cosine',):
        self.embed_module = embed_module
        self.metric = metric
        self.game_words = game_words

    def _get_embeddings_list(self, game_words):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embeddings = sess.run([self.embed_module(game_words)])[0]
        return np.array(embeddings).tolist()

    def _get_nearest(self, hint_embedding, embeddings, game_words, k):
        df = pd.DataFrame(embeddings, index=game_words)
        nn = NearestNeighbors(metric=self.metric).fit(df)
        distance, idx = nn.kneighbors(hint_embedding, n_neighbors=k)
        return distance, idx

    def remove_words(self, words):
        if isinstance(words, str): # if one word
            self.game_words.remove(words)
        else:
            for word in words:
                self.game_words.remove(word)

    def recommend_guess(hint, game_words=None, k=None):
        if game_words is None:
            game_words = self.game_words
        else:
            self.game_words = game_words
        k = len(game_words) if k is None else k

        print('\nGame Words: {}'.format(game_words))
        print('Hint: {}'.format(hint))
        words = [hint] + game_words

        embeddings = self._get_embeddings_list(words)
        hint_embedding, embeddings = embeddings[0], embeddings[1:]
        distance, idx = self._get_nearest(hint_embedding, embeddings, game_words, k)
        print('Recommendations:')
        for i, d in zip(idx, dist):
            print("{:>}: {:.4f}".format(game_words[i], d)) # might need dist[0]

    def ngram_recommend_guess(hint, n_gram=2, game_words=None, k=15):
        """Ngram is a misnomer here. Oh well."""
        # only works with USE
        # use vector addition for Word2Vec?
        if game_words is None:
            game_words = self.game_words
        else:
            self.game_words = game_words

        print('\nGame Words: {}'.format(game_words))
        print('Hint: {}'.format(hint))
        perm = permutations(game_words, n_gram)
        n_grams = [" ".join(x) for x in perm]
        words = [hint] + n_grams

        embeddings = self._get_embeddings_list(words)
        hint_embedding, embeddings = embeddings[0], embeddings[1:]
        distance, idx = self._get_nearest(hint_embedding, embeddings, n_grams, k)
        print('Recommendations:')
        for i, d in zip(idx, dist):
            print("{:>}: {:.4f}".format(game_words[i], d)) # might need dist[0]

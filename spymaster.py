# For spymaster AI, doesn't need to hide or shuffle words, just define red/blue/neutral/assassin at the start, and put team color in explicitly
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

class SpyMasterAI:
    def __init__(self, embed_module, vocabulary_embeddings_path, red_words, blue_words, nuetral_words, assasin_word, metric='cosine'):
        self.embed_module = embed_module
        self.red_words = red_words
        self.blue_words = blue_words
        self.neutral_words = neutral_words
        self.assassin_word = assassin_word
        self.vocabulary_embeddings_path = vocabulary_embeddings_path
        self.vocabulary_embeddings = self.load_embeddings()
        self._remove_Game_vocabulary()
        self.vocabulary = self.vocabulary_embeddings.keys()
        self.metric = metric


    def load_embeddings(self):
        if os.path.exists(self.vocabulary_embeddings_path):
            with open(self.vocabulary_embeddings_path, 'r') as f:
                return json.load(f)
        else: return {}


    def _remove_game_vocabulary(self):
        """removes any words in Game from vocabulary"""
        for word in (self.red-words + self.blue_words + [self.assassin_word]):
            try:
                self.vocabulary_embeddings.pop(word)
            except KeyError:
                continue


    def _get_embeddings_list(self, game_words):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embeddings = sess.run([self.embed_module(game_words)])[0]
        return np.array(embeddings).tolist()


    def _get_nearest(self, phrase_embedding, k=10):
        embeddings = list(self.vocabulary_embeddings.values()) # list of lists, should be sorted the same as self.vocabulary
        # df = pd.DataFrame(self.vocabulary_embeddings, index=self.vocabulary)
        nn = NearestNeighbors(metric=self.metric).fit(embeddings)
        dist, idx = nn.kneighbors(phrase_embedding, n_neighbors=k)
        return dist, idx


    def remove_words(self, words, word_list):
        if word_list == 'red':
            word_list = self.red_words
        elif word_list == 'blue':
            word_list = self.blue_words
        elif word_list == 'nuetral':
            word_list = self.blue_words
        else:
            raise Exception('word_list must be "red", "blue", or "neutral"')
        if isinstance(words, str): # if one word
            word_list.remove(words)
        else:
            for word in words:
                word_list.remove(word)

    def remove_blue(self, words):
        self.remove_words(words, 'blue')

    def remove_red(self, words):
        self.remove_words(words, 'red')

    def remove_neutral(self, words):
        self.remove_words(words, 'neutral')

    def print_word_lists(self):
        print('-'*50)
        print('| Neutral words |')
        print(self.neutral_words)
        print('-'*50)
        print('| Red words |')
        print(self.red_words)
        print('-'*50)
        print('| Blue words |')
        print(self.blue_words)
        print('-'*50)
        print('| Assassin word |')
        print(self.assassin_word)
        print('-'*50)

    def recommend_hint(self, team, team_words=None, problem_words=None, k=10):
        """Recommend hint based on semantic similarity of team words or disimmilarity with non-team words

        Args:
            team: Must be 'red' or 'blue'
            team_words: list of subset of team words to recommend a hint for, if None, uses full list of team words
            k: number of hints to show per embeddings approach
        """
        if team == 'red':
            if team_words:
                team_phrase = " ".join(team_words)
            else:
                team_phrase = " ".join(self.red_words)
            enemy_phrase = " ".join(self.blue_words)
        elif team == 'blue':
            if team_words:
                team_phrase = " ".join(team_words)
            else:
                team_phrase = " ".join(self.blue_words)
            enemy_phrase = " ".join(self.red_words)
        else:
            raise Exception('team must be "red" or "blue"')

        neutral_phrase = " ".join(self.neutral_words)

        # get vectors for k nearest neigbors
        if problem_words:
            problem_phrase = " ".join(problem_words)
            phrase_list = [team_phrase, enemy_phrase, neutral_phrase, self.assassin_word, problem_phrase]
        else:
            phrase_list = [team_phrase, enemy_phrase, neutral_phrase, self.assassin_word]

        embeddings = self._get_embeddings_list(phrase_list)

        if problem_words:
            team_embed, enemy_embed, neutral_embed, assassin_embed, problem_embed = map(np.array, embeddings)
        else:
            team_embed, enemy_embed, neutral_embed, assassin_embed = map(np.array, embeddings)

        # Sentence Encodings are ~ normalized, so vector subtraction should work?
        # may need to change from 0-1 to [-1,1]
        team_minus_assassin = team_embed - assassin_embed
        team_minus_enemies = team_embed - (enemy_embed + assassin_embed)
        team_minus_all = team_embed - (enemy_embed + neutral_embed + assassin_embed)

        if problem_words:
            team_minus_problem = team_embed - problem_embed
            team_embeddings = [team_embed, team_minus_problem, team_minus_assassin, team_minus_enemies, team_minus_all]
            team_embeddings_names = ['team_words', 'team - problem', 'team - assassin', 'team - (enemy+assassin)', 'team - (neutral+enemy+assassin)']

        else:
            team_embeddings = [team_embed, team_minus_assassin, team_minus_enemies, team_minus_all]
            team_embeddings_names = ['team_words', 'team - assassin', 'team - (enemy+assassin)', 'team - (neutral+enemy+assassin)']

        # distance and idx are list of lists
        distance, idx = self._get_nearest(team_embeddings, k)

        print(distance.shape)

        print('| Neutral words |')
        print(neutral_phrase)
        print('-'*50)
        print('| Enemy words |')
        print(enemy_phrase)
        print('-'*50)
        print('| Assassin word |')
        print(self.assassin_word)
        print('-'*50)
        print('| Team words |')
        print(team_phrase)
        print('-'*50)
        if problem_words:
            print('| Problem words |')
            print(problem_phrase)
            print('-'*50)

        for n, name in enumerate(team_embeddings_names):
            print('-'*50)
            print('| {} |'.format(name))
            print('-'*50)
            for i, d in zip(idx[n], distance[n]):
                print("{:>}: {:.4f}".format(self.vocabulary[i], d))
            print('-'*50)

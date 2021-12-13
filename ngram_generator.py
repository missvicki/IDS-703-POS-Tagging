# Generate text Using N Language Model With Brown Corpus

import nltk
from nltk.corpus import brown
import numpy as np

import random
import re
from sys import prefix

corp = brown.words()
corp = [word.lower() for word in corp]
corp


def finish_sentence(sentence, n, deterministic=False):
    """Abstraction to select words based on their probability of occurrence"""
    prefix = sentence

    # get each word from brown corpus
    corpus = brown.words()
    corpus = [word.lower() for word in corpus]
    words = [word for word in corpus]

    # build n-gram and n+1-gram
    n_gram = [corpus[i : i + n - 1] for i in range(len(words) - n + 1)]
    n_1_gram = [corpus[i : i + n] for i in range(len(corpus) - n)]

    # build vocabulary
    vocabulary = sorted(set(words[n - 1 :]))
    token_indices = {char: idx for idx, char in enumerate(vocabulary)}
    vocab_size = len(vocabulary)

    for i in range(25):
        if prefix[-n:] in n_1_gram:  # if n-1 sections of sentence is in n+1-gram
            n_gram_arr = np.zeros(vocab_size)
            for word in n_1_gram:
                if word[:-1] == prefix[-n + 1 :]:
                    n_gram_arr[token_indices[word[-1]]] += 1
            n_gram_arr_norm = n_gram_arr / np.sum(n_gram_arr, axis=0, keepdims=True)
            if deterministic == True:
                next_idx_i = np.where(n_gram_arr_norm == np.amax(n_gram_arr_norm))[0]
                next_idx = min([words.index(vocabulary[i]) for i in next_idx_i])
                prefix.append(words[next_idx])
            else:
                next_word = random.choices(vocabulary, n_gram_arr_norm)
                prefix.append(next_word[0])
            if prefix[-1] in ".?!":
                return prefix
        else:  # if provided sentence is not in n+1 gram execute stupid backoff
            n_1_gram_arr = np.zeros(vocab_size)
            for word in n_gram:
                if word[:-1] == prefix[-n + 2 :]:
                    n_1_gram_arr[token_indices[word[-1]]] += 1
            n_1_gram_arr_norm = n_1_gram_arr / np.sum(
                n_1_gram_arr, axis=0, keepdims=True
            )
            if deterministic == True:
                next_idx_i = np.where(n_1_gram_arr_norm == np.amax(n_1_gram_arr_norm))[
                    0
                ]
                next_idx = min([words.index(vocabulary[i]) for i in next_idx_i])
                prefix.append(words[next_idx])
            else:
                next_word = random.choices(vocabulary, n_1_gram_arr_norm)
                prefix.append(next_word[0])
            if prefix[-1] in ".?!":
                return prefix
    return prefix


print(finish_sentence(["can", "a", "dog"], 3, False))

print(finish_sentence(["The", "current", "state"], 3, False))

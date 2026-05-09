import numpy as np
import re
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import random
from collections import defaultdict, Counter

# ---------- 1. DATA LOADING ----------
def load_data():
    pos = [(movie_reviews.raw(fileid), 1) for fileid in movie_reviews.fileids('pos')]
    neg = [(movie_reviews.raw(fileid), 0) for fileid in movie_reviews.fileids('neg')]
    data = pos + neg
    random.shuffle(data)
    return data

# ---------- 2. PREPROCESSING ----------
def process_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'[^a-z\s]', '', tweet)
    tokens = tweet.split()
    return tokens

# ---------- 3. TRAIN-TEST SPLIT ----------
data = load_data()
split = int(len(data)*0.8)
train = data[:split]
test = data[split:]

# ---------- 4. BUILD FREQUENCY DICTIONARY ----------
def build_freqs(data):
    freqs = defaultdict(int)
    for text, label in data:
        tokens = process_tweet(text)
        for word in tokens:
            freqs[(word, label)] += 1
    return freqs

freqs = build_freqs(train)

# ---------- 5. TRAINING ----------
def train_naive_bayes(freqs, train_data):
    loglikelihood = {}
    vocab = set([word for word, _ in freqs.keys()])
    V = len(vocab)

    pos_words = sum([freqs[(w,1)] for w in vocab])
    neg_words = sum([freqs[(w,0)] for w in vocab])

    logprior = np.log(len([t for t in train_data if t[1]==1]) /
                       len([t for t in train_data if t[1]==0]))

    for word in vocab:
        p_w_pos = (freqs[(word,1)] + 1) / (pos_words + V)
        p_w_neg = (freqs[(word,0)] + 1) / (neg_words + V)

        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, train)

# ---------- 6. PREDICTION ----------
def predict(text):
    words = process_tweet(text)
    p = logprior
    for word in words:
        if word in loglikelihood:
            p += loglikelihood[word]
    return 1 if p > 0 else 0

# ---------- 7. EVALUATION ----------
def accuracy(test):
    correct = 0
    for text, label in test:
        if predict(text) == label:
            correct += 1
    return correct / len(test)

print("Accuracy:", accuracy(test))

# ---------- 8. TEST YOUR OWN TWEET ----------
print(predict("I love this movie"))
print(predict("This is terrible"))

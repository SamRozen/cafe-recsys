# -*- coding: utf-8 -*-
import json
import re
import string
import sys

from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


INPUT_LANG = 'english'


def read_input(filename):
    with open(filename, 'r') as f:
        return json.load(f)

_re_no_punct = re.compile(
    "(%s)" % "|".join(map(re.escape, string.punctuation)))


def get_tokens(text):
    lowers = text.lower().encode('ascii', 'ignore')
    no_punctuation = _re_no_punct.sub('', lowers)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def top_n(tokens, n):
    count = Counter(tokens)
    return count.most_common(n)


def filter_tokens(tokens):
    return [w for w in tokens if w not in stopwords.words(INPUT_LANG)]


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    t = filter_tokens(get_tokens(text))
    stemmer = PorterStemmer()
    return stem_tokens(t, stemmer)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python %s <filename> <top_n>' % (
            sys.argv[0])
        raise ValueError("Invalid command line parameters")

    filename = sys.argv[1]
    n = int(sys.argv[2])

    data = read_input(filename)
    stemmer = PorterStemmer()
    tokens = {}
    for item in data:
        t = filter_tokens(get_tokens(item['description']))
        key = '%s - %s' % (item['brand'], item['name'])
        tokens[key] = stem_tokens(t, stemmer)
    for k, v in tokens.iteritems():
        print k, top_n(v, n)

    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    texts = [item['description'] for item in data]
    keys = [item['name'] for item in data]
    tfs = tfidf.fit_transform(texts)

    dist = 1 - pairwise_distances(tfs, metric='cosine')
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            print 'Sim(%s, %s) : %f' % (keys[i], keys[j], dist[i][j])

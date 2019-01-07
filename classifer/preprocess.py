from __future__ import unicode_literals

import collections
import io
import re

import numpy
import progressbar


split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')


def split_sentence(s):
    s = s.lower()
    s = s.replace('\u2019', "'")
    s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        words.extend(split_pattern.split(word))
    words = [w for w in words if w]
    return words


def open_file(path):
    return io.open(path,encoding="utf-8", errors='ignore')


def count_lines(path):
    with open_file(path) as f:
        return sum([1 for _ in f])


def read_file(path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    max_value=n_lines
	
    with open_file(path) as f:
        for line in bar(f, max_value):
            words = split_sentence(line)
            yield words


def count_words(path, max_vocab_size=40000):
    counts = collections.Counter()
    for words in read_file(path):
        for word in words:
            counts[word] += 1

    vocab = [word for (word, _) in counts.most_common(max_vocab_size)]
    return vocab


def make_dataset(path, vocab):
    word_id = {word: index for index, word in enumerate(vocab)}
    dataset = []
    for words in read_file(path):
        array = make_array(word_id, words)
        dataset.append(array)
    return dataset


def make_array(word_id, words):
    ids = [word_id.get(word, 1) for word in words]
    return ids

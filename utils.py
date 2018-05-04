""" utility functions"""
import re
import os

def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


PAD = 0
UNK = 1
START = 2
END = 3
def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id

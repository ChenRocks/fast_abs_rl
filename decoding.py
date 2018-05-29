""" decoding utilities"""
import json
import re
import os
from os.path import join
import pickle as pkl
from itertools import starmap

from cytoolz import curry

import torch

from utils import PAD, UNK, START, END
from model.copy_summ import CopySumm
from model.extract import ExtractSumm, PtrExtractSumm
from model.rl import ActorCritic
from data.batcher import conver2id, pad_batch_tensorize
from data.data import CnnDmDataset


try:
    DATASET_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class DecodeDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        assert split in ['val', 'test']
        super().__init__(split, DATASET_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return art_sents


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def load_best_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0]))
    )['state_dict']
    return ckpt


class Abstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'base_abstractor'
        abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(abs_dir)
        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
        abstractor = CopySumm(**abs_args)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len

    def _prepro(self, raw_article_sents):
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        for raw_words in raw_article_sents:
            for w in raw_words:
                if not w in ext_word2id:
                    ext_word2id[w] = len(ext_word2id)
                    ext_id2word[len(ext_id2word)] = w
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        art_lens = [len(art) for art in articles]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
        extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                        ).to(self._device)
        extend_vsize = len(ext_word2id)
        dec_args = (article, art_lens, extend_art, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

    def __call__(self, raw_article_sents):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        decs, attns = self._net.batch_decode(*dec_args)
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        dec_sents = []
        for i, raw_words in enumerate(raw_article_sents):
            dec = []
            for id_, attn in zip(decs, attns):
                if id_[i] == END:
                    break
                elif id_[i] == UNK:
                    dec.append(argmax(raw_words, attn[i]))
                else:
                    dec.append(id2word[id_[i].item()])
            dec_sents.append(dec)
        return dec_sents


class BeamAbstractor(Abstractor):
    def __call__(self, raw_article_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        dec_args = (*dec_args, beam_size, diverse)
        all_beams = self._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word),
                                 zip(all_beams, raw_article_sents)))
        return all_beams

@curry
def _process_beam(id2word, beam, art_sent):
    def process_hyp(hyp):
        seq = []
        for i, attn in zip(hyp.sequence[1:], hyp.attns[:-1]):
            if i == UNK:
                art_sent[max(range(len(art_sent)),
                             key=lambda j: attn[j].item())]
            else:
                seq.append(id2word[i])
        hyp.sequence = seq
        del hyp.hists
        del hyp.attns
        return hyp
    return list(map(process_hyp, beam))


class Extractor(object):
    def __init__(self, ext_dir, max_ext=5, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        if ext_meta['net'] == 'ml_ff_extractor':
            ext_cls = ExtractSumm
        elif ext_meta['net'] == 'ml_rnn_extractor':
            ext_cls = PtrExtractSumm
        else:
            raise ValueError()
        ext_ckpt = load_best_ckpt(ext_dir)
        ext_args = ext_meta['net_args']
        extractor = ext_cls(**ext_args)
        extractor.load_state_dict(ext_ckpt)
        word2id = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = extractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_ext = max_ext

    def __call__(self, raw_article_sents):
        self._net.eval()
        n_art = len(raw_article_sents)
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        indices = self._net.extract([article], k=min(n_art, self._max_ext))
        return indices


class ArticleBatcher(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')

    def __call__(self, raw_article_sents):
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        return article

class RLExtractor(object):
    def __init__(self, ext_dir, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        assert ext_meta['net'] == 'rnn-ext_abs_rl'
        ext_args = ext_meta['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(ext_dir, 'agent_vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(word2id, cuda))
        ext_ckpt = load_best_ckpt(ext_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = agent.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}

    def __call__(self, raw_article_sents):
        self._net.eval()
        indices = self._net(raw_article_sents)
        return indices

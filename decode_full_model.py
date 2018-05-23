""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time

from cytoolz import identity

import torch
from torch.utils.data import DataLoader

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset
from decoding import make_html_safe


# FIXME
MAX_ART_LEN = 200  # TODO
MAX_ART_NUM = 60

MAX_ABS_NUM = 6  # need to set max sentences to extract for non-RL extractor


def decode(save_path, model_dir, split, batch_size, beam_size, max_len, cuda):
    if beam_size != 1:
        raise NotImplementedError
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        abstractor = Abstractor(join(model_dir, 'abstractor'), max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    save_path = join(save_path, split)
    os.makedirs(join(save_path, 'output'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            ext_arts = []
            ext_inds = []
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += list(map(lambda i: raw_art_sents[i.item()], ext))
            dec_outs = abstractor(ext_arts)
            assert i == batch_size*i_debug
            if beam_size > 1:
                # TODO
                pass  # reranking model
            else:
                for j, n in ext_inds:
                    decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                    with open(join(save_path, 'output/{}.dec'.format(i)),
                              'w') as f:
                        f.write(make_html_safe('\n'.join(decoded_sents)))
                    i += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        i, n_data, i/n_data*100,
                        timedelta(seconds=int(time()-start))
                    ), end='')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # data
    parser.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.max_dec_word, args.cuda)

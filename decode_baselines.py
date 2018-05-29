""" run decoding of X-ext (+ abs)"""
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

from decoding import Abstractor, Extractor, DecodeDataset
from decoding import make_html_safe


MAX_ABS_NUM = 6  # need to set max sentences to extract for non-RL extractor


def decode(save_path, abs_dir, ext_dir, split, batch_size, max_len, cuda):
    start = time()
    # setup model
    if abs_dir is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        abstractor = identity
    else:
        abstractor = Abstractor(abs_dir, max_len, cuda)
    if ext_dir is None:
        # NOTE: if no abstractor is provided then
        #       it would be  the lead-N extractor
        extractor = lambda art_sents: list(range(len(art_sents)))[:MAX_ABS_NUM]
    else:
        extractor = Extractor(ext_dir, max_ext=MAX_ABS_NUM, cuda=cuda)

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
    for i in range(MAX_ABS_NUM):
        os.makedirs(join(save_path, 'output_{}'.format(i)))
    dec_log = {}
    dec_log['abstractor'] = (None if abs_dir is None
                             else json.load(open(join(abs_dir, 'meta.json'))))
    dec_log['extractor'] = (None if ext_dir is None
                            else json.load(open(join(ext_dir, 'meta.json'))))
    dec_log['rl'] = False
    dec_log['split'] = split
    dec_log['beam'] = 1  # greedy decoding only
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
                ext = extractor(raw_art_sents)
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += list(map(lambda i: raw_art_sents[i], ext))
            dec_outs = abstractor(ext_arts)
            assert i == batch_size*i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                for k, dec_str in enumerate(decoded_sents):
                    with open(join(save_path, 'output_{}/{}.dec'.format(k, i)),
                              'w') as f:
                        f.write(make_html_safe(dec_str))

                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100, timedelta(seconds=int(time()-start))
                ), end='')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset')
    )
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--abs_dir', help='root of the abstractor model')
    parser.add_argument('--ext_dir', help='root of the extractor model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--n_ext', type=int, action='store', default=4,
                        help='number of sents to be extracted')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.abs_dir, args.ext_dir,
           data_split, args.batch, args.max_dec_word, args.cuda)

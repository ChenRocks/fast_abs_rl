""" train extractor (ML)"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.extract import ExtractSumm, PtrExtractSumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer

from utils import PAD, UNK
from utils import make_vocab, make_embedding

from data.data import CnnDmDataset
from data.batcher import coll_fn_extract, prepro_fn_extract
from data.batcher import convert_batch_extract_ff, batchify_fn_extract_ff
from data.batcher import convert_batch_extract_ptr, batchify_fn_extract_ptr
from data.batcher import BucketedGenerater


BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class ExtractDataset(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted']
        return art_sents, extracts


def build_batchers(net_type, word2id, cuda, debug):
    assert net_type in ['ff', 'rnn']
    prepro = prepro_fn_extract(args.max_word, args.max_sent)
    def sort_key(sample):
        src_sents, _ = sample
        return len(src_sents)
    batchify_fn = (batchify_fn_extract_ff if net_type == 'ff'
                   else batchify_fn_extract_ptr)
    convert_batch = (convert_batch_extract_ff if net_type == 'ff'
                     else convert_batch_extract_ptr)
    batchify = compose(batchify_fn(PAD, cuda=cuda),
                       convert_batch(UNK, word2id))

    train_loader = DataLoader(
        ExtractDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        ExtractDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher


def configure_net(net_type, vocab_size, emb_dim, conv_hidden,
                  lstm_hidden, lstm_layer, bidirectional):
    assert net_type in ['ff', 'rnn']
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['conv_hidden']   = conv_hidden
    net_args['lstm_hidden']   = lstm_hidden
    net_args['lstm_layer']    = lstm_layer
    net_args['bidirectional'] = bidirectional

    net = (ExtractSumm(**net_args) if net_type == 'ff'
           else PtrExtractSumm(**net_args))
    return net, net_args


def configure_training(net_type, opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    assert net_type in ['ff', 'rnn']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    if net_type == 'ff':
        criterion = lambda logit, target: F.binary_cross_entropy_with_logits(
            logit, target, reduce=False)
    else:
        ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
        def criterion(logits, targets):
            return sequence_loss(logits, targets, ce, pad_idx=-1)

    return criterion, train_params


def main(args):
    assert args.net_type in ['ff', 'rnn']
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(args.net_type, word2id,
                                                args.cuda, args.debug)

    # make net
    net, net_args = configure_net(args.net_type,
                                  len(word2id), args.emb_dim, args.conv_hidden,
                                  args.lstm_hidden, args.lstm_layer, args.bi)
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding(
            {i: w for w, i in word2id.items()}, args.w2v)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        args.net_type, 'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'ml_{}_extractor'.format(args.net_type)
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the feed-forward extractor (ff-ext, ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--net-type', action='store', default='rnn',
                        help='model type of the extractor (ff/rnn)')
    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100,
                        help='the number of hidden units of Conv')
    parser.add_argument('--lstm_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of lSTM')
    parser.add_argument('--lstm_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM Encoder')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)

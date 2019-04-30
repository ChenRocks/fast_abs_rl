import argparse
import pickle as pkl
import os
from os.path import join, exists
from types import SimpleNamespace

import torch
from train_abstractor import prep_trainer as abs_prep_trainer
from train_extractor_ml import prep_trainer as exs_prep_trainer
from utils import make_vocab, make_embedding

try:
  DATA_DIR = os.environ['DATA']
except KeyError:
  print('please use environment variable to specify datadirectories')

def main(args):
  with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
      wc = pkl.load(f)
      word2id = make_vocab(wc, args.vsize)

  abs_args = SimpleNamespace(
    **vars(args),
    path='./uni_pretrained_abstractor',
    w2v='./word_vectors/word2vec.128d.226k.bin',
    n_layer=1,
    n_hidden=256,
    max_art=100,
    max_abs=30,
  )
  abs_trainer, abs_net = abs_prep_trainer(abs_args, word2id=word2id)

  exs_args = SimpleNamespace(
    **vars(args),
    path='./uni_pretrained_extractor',
    w2v=None, # no embedding since reuse abs's encoder
    net_type='rnn',
    lstm_layer=1,
    lstm_hidden=256,
    max_word=100,
    max_sent=60
  )

  exs_trainer, _ = exs_prep_trainer(exs_args, word2id=word2id, encoder=abs_net.encoder)

  exs_trainer.train()
  abs_trainer.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='training of the abstractor (ML)'
  )

  parser.add_argument('--vsize', type=int, action='store', default=30000,
                      help='vocabulary size')
  parser.add_argument('--emb_dim', type=int, action='store', default=128,
                      help='the dimension of word embedding')


  parser.add_argument('--no-bi', action='store_true',
                      help='disable bidirectional LSTM encoder')

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

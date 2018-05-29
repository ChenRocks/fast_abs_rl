""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import re
import os
from os.path import join, exists

from evaluate import eval_meteor, eval_rouge


try:
    _DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

def make_summaries(decode_dir, n_ext):
    out_dir = join(decode_dir, 'output_top{}'.format(args.n_ext))
    os.makedirs(out_dir)
    decs = os.listdir(join(decode_dir, 'output_0'))
    dec_matcher = re.compile('[0-9]*.dec')
    decs = sorted([d for d in decs if dec_matcher.match(d)],
                   key=lambda d: float(d.split('.')[0]))
    for d in decs:
        sents = []
        for i in range(n_ext):
            fname = join(decode_dir, 'output_{}/{}'.format(i, d))
            if exists(fname):
                with open(fname, 'r') as f:
                    sents.append(f.read())
        with open(join(out_dir, d), 'w') as f:
            f.write('\n'.join(sents))


def main(args):
    dec_dir = join(args.decode_dir, 'output_top{}'.format(args.n_ext))
    if not exists(dec_dir):
        make_summaries(args.decode_dir, args.n_ext)
    with open(join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = join(_DATA_DIR, 'refs', split)
    assert exists(ref_dir)

    if args.rouge:
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'rouge'
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
    print(output)
    with open(join(args.decode_dir,
                   'top{}_{}.txt'.format(args.n_ext, metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the baseline models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')

    parser.add_argument('--n_ext', type=int, action='store', required=True,
                        help='number of sentences to extract')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')

    args = parser.parse_args()
    main(args)

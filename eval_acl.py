""" Evaluate the output files to get the numbers reported in ACL18"""
import argparse
from os.path import join, abspath, dirname, exists

from evaluate import eval_meteor, eval_rouge


_REF_DIR = join(abspath(dirname(__file__)), 'acl18_results')


def main(args):
    dec_dir = args.decode_dir
    ref_dir = join(_REF_DIR, 'reference')
    if args.rouge:
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
    print(output)


if __name__ == '__main__':
    assert exists(_REF_DIR)
    parser = argparse.ArgumentParser(
        description='Evaluate the output files to get the numbers reported'
                    ' as in the ACL paper'
    )

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')

    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')

    args = parser.parse_args()
    main(args)

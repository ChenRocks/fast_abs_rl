""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat

import torch
import torch.multiprocessing as mp


# Batching functions
def coll_fn(data):
    source_lists, target_lists = unzip(data)
    sources = list(filter(bool, source_lists))
    targets = list(filter(bool, target_lists))
    assert all(sources) and all(targets)
    return sources, targets


def tokenize(unk, word2id, max_len, texts):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in t.lower().split()[:max_len]] for t in texts]

@curry
def prepro_fn(unk, word2id, max_src_len, max_tgt_len, batch):
    sources, targets = batch
    ext_word2id = dict(word2id)
    for words in sources:
        for w in words:
            if not w in ext_word2id:
                ext_word2id[w] = len(ext_word2id)
    src_exts = tokenize(unk, ext_word2id, max_src_len, concat(sources))
    sources = tokenize(unk, word2id, max_src_len, concat(sources))
    tar_ins = tokenize(unk, word2id, max_tgt_len, concat(targets))
    targets = tokenize(unk, ext_word2id, max_tgt_len, concat(targets))
    batch = list(zip(sources, src_exts, tar_ins, targets))
    return batch


def pad_batch_tensorize(inputs, pad=0):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    batch_size = len(inputs)
    max_len = max(arr.size(0) for arr in inputs)
    # assuming equal trailing dimensions across list
    arr_shape = tuple(list(inputs[0].size())[1:])
    tensor_shape = (batch_size, max_len) + arr_shape
    tensor = inputs[0].new(*tensor_shape)
    tensor.fill_(pad)
    for i, arr in enumerate(inputs):
        tensor[i, :arr.size(0)] = arr
    return tensor

@curry
def batchify_fn(pad, start, end, data, cuda=True):
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(data)))
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor

    src_lens = [len(src) for src in sources]
    sources = [tensor_type(src) for src in sources]
    ext_srcs = [tensor_type(ext) for ext in ext_srcs]

    tar_ins = [tensor_type([start] + tgt) for tgt in tar_ins]
    targets = [tensor_type(tgt + [end]) for tgt in targets]

    source = pad_batch_tensorize(sources, pad)
    tar_in = pad_batch_tensorize(tar_ins, pad)
    target = pad_batch_tensorize(targets, pad)
    ext_src = pad_batch_tensorize(ext_srcs, pad)

    ext_vsize = ext_src.max().item()
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args


def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            hyper_batch.sort(key=self._sort_key)
            indexes = list(range(0, len(hyper_batch), batch_size))
            random.shuffle(indexes)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(target=_batch2q,
                                        args=(self._loader, self._prepro,
                                              self._queue, self._single_run))
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()

# Fast Abstractive Summarization-RL
This repository contains the code for our ACL 2018 paper:

*[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/abs/1805.11080)*.

You can
1. Look at the generated summaries and evaluate the ROUGE/METEOR scores
2. Run decoding of the pretrained model
3. Train your own model

If you use this code, please cite our paper:
```
@inproceedings{chen2018fast,
  title={Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting},
  author={Yen-Chun Chen and Mohit Bansal},
  booktitle={Proceedings of ACL},
  year={2018}
}
```

## Dependencies
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 0.4.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system.

## Evaluate the output summaries from our ACL paper
Download the output summaries *[here](https://bit.ly/acl18_results)*.

To evaluate, you will need to download and setup the official ROUGE and METEOR
packages.

We use [`pyrouge`](https://github.com/bheinzerling/pyrouge)
(`pip install pyrouge` to install)
to make the ROUGE XML files required by the official perl script.
You will also need the official ROUGE package.
(However, it seems that the original ROUGE website is down.
An alternative can be found
*[here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)*.)
Please specify the path to your ROUGE package by setting the environment variable
`export ROUGE=[path/to/rouge/directory]`.


For METEOR, we only need the JAR file `meteor-1.5.jar`.
Please specify the file by setting the environment variable
`export METEOR=[path/to/meteor/jar]`.

Run
```
python eval_acl.py --[rouge/meteor] --decode_dir=[path/to/decoded/files]
```
to get the ROUGE/METEOR scores reported in the paper.

## Decode summaries from the pretrained model
Download the pretrained models *[here](https://bit.ly/acl18_pretrained)*.
You will also need a preprocessed version of the CNN/DailyMail dataset.
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.
After that, specify the path of data files by setting the environment variable
`export DATA=[path/to/decompressed/data]`

We provide 2 versions of pretrained models.
Using `acl` you can reproduce the results reported in our paper.
Using `new` you will get our latest result trained with a newer version of PyTorch library
which leads to slightly higher scores.

To decode, run
```
python decode_full_model.py --path=[path/to/save/decoded/files] --model_dir=[path/to/pretrained] --beam=[beam_size] [--test/--val]
```
Options:
- beam_size: number of hypothesis for (diverse) beam search. (use beam_size > 1 to enable reranking)
  - beam_szie=1 to get greedy decoding results (rnn-ext + abs + RL)
  - beam_size=5 is used in the paper for the +rerank model (rnn-ext + abs + RL + rerank)
- test/val: decode on test/validation dataset

If you want to evaluate on the generated output files,
please follow the instructions in the above section to setup ROUGE/METEOR.

Next, make the reference files for evaluation:
```
python make_eval_references.py
```
and then run evaluation by:
```
python eval_full_model.py --[rouge/meteor] --decode_dir=[path/to/save/decoded/files]
```

### Results
You should get the following results

Validation set

| Models             | ROUGEs (R-1, R-2, R-L) | METEOR |
| ------------------ |:----------------------:| ------:|
| **acl** |
| rnn-ext + abs + RL | (41.01, 18.20, 38.57)  |  21.10 |
| + rerank           | (41.74, 18.39, 39.41)  |  20.45 |
| **new** |
| rnn-ext + abs + RL | (41.23, 18.45, 38.71)  |  21.14 |
| + rerank           | (42.07, 18.81, 39.69)  |  20.58 |

Test set

| Models             | ROUGEs (R-1, R-2, R-L) | METEOR |
| ------------------ |:----------------------:| ------:|
| **acl** |
| rnn-ext + abs + RL | (40.03, 17.61, 37.58)  |  21.00 |
| + rerank           | (40.88, 17.81, 38.54)  |  20.39 |
| **new** |
| rnn-ext + abs + RL | (40.41, 17.92, 37.87)  |  21.13 |
| + rerank           | (41.20, 18.18, 38.79)  |  20.55 |

**NOTE**:
The original models in the paper are trained with pytorch 0.2.0 on python 2. 
After the acceptance of the paper, we figured it is better for the community if
we release the code with latest libraries so that it becomes easier to build new
models/techniques on top of our work. 
This results in a negligible difference w.r.t. our paper results when running the old pretrained model;
and gives slightly better scores than our paper if running the new pretrained model.

## Train your own models
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.
After that, specify the path of data files by setting the environment variable
`export DATA=[path/to/decompressed/data]`

To re-train our best model:
1. pretrained a *word2vec* word embedding
```
python train_word2vec.py --path=[path/to/word2vec]
```
2. make the pseudo-labels
```
python make_extraction_labels.py
```
3. train *abstractor* and *extractor* using ML objectives
```
python train_abstractor.py --path=[path/to/abstractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
python train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
```
4. train the *full RL model*
```
python train_full_rl.py --path=[path/to/save/model] --abs_dir=[path/to/abstractor/model] --ext_dir=[path/to/extractor/model]
```
After the training finishes you will be able to run the decoding and evaluation following the instructions in the previous section.

The above will use the best hyper-parameters we used in the paper as default.
Please refer to the respective source code for options to set the hyper-parameters.


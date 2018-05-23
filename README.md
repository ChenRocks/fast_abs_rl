# Fast Abstractive Summarization-RL
This repository contains the code for our ACL 2018 paper:

<!--- TODO -->
*[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](arxiv.org/abs/xxxxx)*.

You can
1. Look at the generated summaries and evaluate the ROUGE/METEOR scores
2. Run decoding of the pretrained model (**WIP**)
3. Train your own model (**WIP**)

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

## Evaluate the output summaries from our ACL paper
Download the output summaries *[here](bitly/acl18_results)*.

To evaluate, you will need to download and setup the official ROUGE and METEOR
packages.

We use [`pyrouge`](https://github.com/bheinzerling/pyrouge)
(`pip install pyrouge` to install)
to make the ROUGE XML files required by the official perl script.
You will also need the offical ROUGE package.
(However, it seems that the original ROUGE website is down.
An alternative can be found
*[here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)*.)
Please specify the path to your ROUGE package by setting the environment variable
`export ROUGE=[path/to/rouge/directory]`.


For METEOR, we only need the JAR file `meteor-1.5.jar`.
Please specify the file by setting the environment variable
`export METEOR=[path/to/meteor/jar]`.

Run `python eval_acl.py --[rouge/meteor] --decode_dir=[path/to/decoded/files]`
to get the ROUGE/METEOR scores reported in the paper.

## Decode summaries from the pretrained model
<!--- TODO -->
*Work in progress...*

## Train your own models
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.

<!--- TODO -->
*Work in progress...*

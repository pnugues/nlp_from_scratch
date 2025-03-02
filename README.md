# nlp_from_scratch
A reimplementation in PyTorch of the Senna program described in [_Natural language processing (almost) from scratch_](https://arxiv.org/abs/1103.0398) by Collobert et al. (2011).

## Overview
In these two notebooks, I reproduce the Senna program described in [_Natural language processing (almost) from scratch_ by Collobert et al. (2011)](https://arxiv.org/abs/1103.0398) with a modern deep-learning programming interface: PyTorch. I created the notebooks from the paper's high-level description. This means that my programs are reinterpretations and are not exactly equivalent to the original code.

Collobert et al. created a set of programs with configurations ranging from a word embeddings input to inputs including specific dictionaries on the word properties. These dictionaries were designed for English, including for instance the word suffixes. Here I will assume a minimal knowledge of the words. As input, I will use the dataset words and capitalization properties, as we can extend them easily to other languages, and the pretrained embeddings.

Collobert et al. proposed two main architectures: One with two linear layers and the other adding a convolutional layer. They used the simpler feed-forward network with two layers for the part-of-speech (POS) tagger, chunker, and named entity recognition and the second network for semantic role labeling. I reimplemented the first network only. To optimize the sequences, Collobert et al. added a conditional random field (CRF) layer to their first system. This corresponds to the two notebooks: The first one with the feed-forward sequence and the second one, where I added the CRF.

Ronan Collobert made the inference part of Senna available as well as the pretrained embeddings. See here: https://ronan.collobert.com/senna/. We will need the embeddings and the word list to run the notebook. To download them, you must first approve the license. Then uncompress the archive and move the folder to your working folder. A possible extension of the notebook would be to replace the embeddings with other vectors such as word2vec or train new ones using [Gensim](https://radimrehurek.com/gensim/).

 I trained them respectively on:
 * A corpus following the Universal Dependencies format for POS tagging. Here I used the [English Web Treebank](https://github.com/UniversalDependencies/UD_English-EWT/tree/master);
 * CoNLL 2000 for chunking;
 * CoNLL 2003 for named entity recognition.

## Experimental Setup
The datasets have different structures. EWT and CoNLL 2003 have a validation set, but CoNLL 2000 does not. Collobert et al. do not provide details on the fitting and evaluation conditions. When CoNLL 2000 was released, experimental protocols were not yet well standardized, which is reflected in its structure. Attardi merges the fitting and validation sets and removes the title lines. 

Collobert et al. used the Penn Treebank for POS tagging, CoNLL 2000, and CoNLL 2003. They did not describe precisely how they selected their model and how (or if) they used the CoNLL 2003 validation set. 

As there is no validation set in CoNLL 2000, in my experiments, I used the test set of both CoNLL 2000 and CoNLL 2003 as validation set and I included the validation set of CoNLL 2003 in the training set. The results I report are then not equivalent to those of Collobert et al. For the EWT, I report the classical training, validation, and test results.

|Dataset|POS EWT | CoNLL 2000|CoNLL 2003|
|-------| -------- | ------- |-------|
|Training|Train|Train|Train + val|
|Validation|Val|Test|Test|
|Test|Test|--|--|

## Results
I report the results of the simple feed-forward network and different configurations (first notebook), then of the CRF network (second notebook). I used 25 epochs in the first notebook and 60 for the second one. 

### Feed-Forward
|Batch|LR| Optimizer | $\epsilon$|Embeddings | Init.|Tags|POS EWT | CoNLL 2000|CoNLL 2003|
|-------:| -------- | ------- |-------|-------|-------|----|----|----|----|
|1| 0.01|Adagrad |$10^{-10}$ | Words    |U centré / 10|BIO|0.9411 (ep. 21)|0.9119 (ep. 14)|0.8219 (ep. 22)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna    |U centré / 10|BIO|0.9468 (ep. 23)|0.9241 (ep. 19)|0.8497 (ep. 13)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna + Words    |U centré / 10|BIO|0.9516 (ep. 21)|0.9242 (ep. 20)|0.8466 (ep. 9)|
|1| 0.01|Adagrad |$10^{-6}$ | Senna + Words    |U centré / 10|BIO|**0.9520** (ep. 22)|**0.9251** (ep. 21)|0.8448 (ep. 20)|
|1| 0.01|Adagrad |$10^{-10}$ | Words    |U centré / 10|IOBES|0.9411 (ep. 21)|0.9061 (ep. 16)|0.8254 (ep. 15)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna    |U centré / 10|IOBES|0.9468 (ep. 23)|0.9207 (ep. 19)|**0.8615** (ep. 15)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna + Words   |U centré / 10 |IOBES|0.9516 (ep. 21)|0.9220 (ep. 13)|0.8614 (ep. 18)|

### Feed-Forward and CRF
|Batch|LR| Optimizer | $\epsilon$|Embeddings | Init.|Tags|POS EWT | CoNLL 2000|CoNLL 2003|
|-------:| -------- | ------- |-------|-------|-------|----|----|----|----|
|1| 0.01|Adagrad |$10^{-10}$ | Words    |U centré / 10|BIO|–|0.9268 (ep. 20)|0.8530 (ep. 11)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna    |U centré / 10|BIO|–|0.9354 (ep. 29)|0.8778 (ep. 25)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna + Words    |U centré / 10|BIO|–|0.9357 (ep. 20)|0.8760 (ep. 21)|
|1| 0.01|Adagrad |$10^{-6}$ | Senna + Words    |U centré / 10|BIO||||
|1| 0.01|Adagrad |$10^{-10}$ | Words    |U centré / 10|IOBES|0.9429 (ep. 26)|0.9261 (ep. 34)|0.8569 (ep. 11)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna    |U centré / 10|IOBES|0.9503 (ep. 13)|0.9367 (ep. 34)| 0.8848 (ep. 30)|
|1| 0.01|Adagrad |$10^{-10}$ | Senna + Words   |U centré / 10 |IOBES|0.9514 (ep. 57)|**0.9383** (ep. 40)|**0.8874** (ep. 44)|

## Senna Results

## Other Implementations
I could find a few other attempts to reproduce the code. To the best of my knowledge, no one used PyTorch.
 * For the taggers and the embeddings, see the excellent [`deepnl`](https://github.com/attardi/deepnl) by Giuseppe Attardi. See also his paper [DeepNL: a Deep Learning NLP pipeline](https://aclanthology.org/W15-1515/); 
 * For the Senna embeddings, see https://github.com/klb3713/cw_word_embedding.

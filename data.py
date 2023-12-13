# import chardet
from tqdm.contrib import tzip


def load_vocab():

    with open('ChineseCorpus199801.txt', 'r', encoding='gb2312', errors='ignore') as f:
        corpus = f.readlines()
    
    # ['19980131-04-004-004/m', '这/r', '就/d', '是/v', '[江/j', '峡/j', '大道/n]ns', '。/w'] 64

    vocab = []
    for c in corpus:
        word_list = c.split(' ')
        word_list = [w for w in word_list if w != '\n']
        word_list = word_list[1:]
        word_list = [w.split('/')[0] for w in word_list]
        word_list = [w.split('[')[0] for w in word_list]
        word_list = [w for w in word_list if len(w)]
        if not len(word_list):
            continue
        vocab.extend(word_list)
    vocab = list(set(vocab))
    
    return vocab


def load_gt_sentences():

    with open('ChineseCorpus199801.txt', 'r', encoding='gb2312', errors='ignore') as f:
        corpus = f.readlines()
    
    gt_sentences = []
    for c in corpus:
        word_list = c.split(' ')
        word_list = [w for w in word_list if len(w)]
        word_list = [w for w in word_list if w != '\n']
        word_list = word_list[1:]
        word_list = [w.split('/')[0] for w in word_list]
        word_list = [w.split('[')[0] for w in word_list]
        if not len(word_list):
            continue
        gt_sentences.append(''.join(word_list))
    
    return gt_sentences


def load_gt_partition():

    with open('ChineseCorpus199801.txt', 'r', encoding='gb2312', errors='ignore') as f:
        corpus = f.readlines()
    
    gt_partition = []
    for c in corpus:
        word_list = c.split(' ')
        word_list = [w for w in word_list if w != '\n']
        word_list = word_list[1:]
        word_list = [w.split('/')[0] for w in word_list]
        word_list = [w.split('[')[0] for w in word_list]
        word_list = [w for w in word_list if len(w)]
        if not len(word_list):
            continue
        gt_partition.append(word_list)
    
    return gt_partition

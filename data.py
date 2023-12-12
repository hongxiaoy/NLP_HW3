import chardet


def load_vocab():

    with open('ChineseCorpus199801.txt', 'r', encoding='gb2312', errors='ignore') as f:
        corpus = f.readlines()

    vocab = []
    for c in corpus:
        word_list = c.split(' ')
        word_list = [w for w in word_list if len(w)]
        word_list = [w for w in word_list if w != '\n']
        word_list = word_list[1:]
        word_list = [w.split('/')[0] for w in word_list]
        word_list = [w.split('[')[0] for w in word_list]
        vocab.extend(word_list)
    vocab = list(set(vocab))
    
    return vocab

# ['19980131-04-004-004/m', '这/r', '就/d', '是/v', '[江/j', '峡/j', '大道/n]ns', '。/w'] 64
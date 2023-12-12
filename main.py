import threading
from tqdm.contrib import tzip

from data import load_gt_partition, load_gt_sentences, load_vocab

from bidirectionalmm import partition as bi_mm
from fmm import partition as f_mm

gt_partition = load_gt_partition()
gt_sentences = load_gt_sentences()
vocab = load_vocab()

total = 0
right = 0
order = 0
for s, p in tzip(gt_sentences, gt_partition):
    order += 1
    pred = bi_mm(s, vocab)
    if p == pred:
        right += 1
    total += 1
    # print(p)
    # print(pred)
    print(f'{order}/{len(gt_sentences)}: {right}/{total}, {right/total}')
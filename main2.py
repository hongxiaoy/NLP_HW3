from functools import partial
import threading
from mmcv import track_progress
from tqdm.contrib import tzip

from data import load_gt_partition, load_gt_sentences, load_vocab

from bmm import partition as b_mm

gt_partition = load_gt_partition()
gt_sentences = load_gt_sentences()
vocab = load_vocab()

total = 0
right = 0
order = 0

def main(sent_part):
    s = sent_part[0]
    p = sent_part[1]
    pred = b_mm(s, vocab)
    if p == pred:
        global right
        right += 1
    global total
    total += 1


#对一系列项目和任务跟踪进度，进度条原位置刷新的方式
track_progress(main, list(zip(gt_sentences, gt_partition)))
print(right, total, right/total)
#并行任务的跟踪进度
# mmcv.track_parallel_progress(func,tasks,nproc)
#刷新位置的进度条方式
# mmcv.track_iter_progress(tasks)
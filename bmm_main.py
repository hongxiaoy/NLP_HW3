from functools import partial
import threading
from mmcv import track_progress
from tqdm.contrib import tzip

from data import load_gt_partition, load_gt_sentences, load_vocab

from bmm import partition as b_mm

gt_partition = load_gt_partition()
gt_sentences = load_gt_sentences()
vocab = load_vocab()

TP = 0
FN = 0
FP = 0
IV = 0
IV_R = 0
OOV = 0
OOV_R = 0

def main(sent_part):
    global TP
    global FN
    global FP
    global IV
    global IV_R
    global OOV
    global OOV_R
    
    s = sent_part[0]
    p = sent_part[1]
    pred = b_mm(s, vocab)

    # 将分词结果转换为区间
    p_number = []
    start = 0
    for w in p:
        end = start + len(w)
        p_number.append((start, end))
        start = end
    p_number = set(p_number)
    # 将分词结果转换为区间
    pred_number = []
    start = 0
    for w in pred:
        end = start + len(w)
        pred_number.append((start, end))
        start = end
    pred_number = set(pred_number)
    
    tp = len(p_number & pred_number)
    TP += tp
    fp = len(pred_number) - tp
    FP += fp
    fn = len(p_number) - tp
    FN += fn

    for (start, end) in p_number:
        word = s[start:end]
        if word in vocab:
            IV += 1
        else:
            OOV += 1
    
    
    for (start, end) in p_number & pred_number:
        word = s[start:end]
        if word in vocab:
            IV_R += 1
        else:
            OOV_R += 1

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    OOV_Recall = OOV_R / (OOV + 1e-6)
    IV_Recall = IV_R / (IV + 1e-6)
    print('\n', Precision, Recall, F1, OOV_Recall, IV_Recall)


#对一系列项目和任务跟踪进度，进度条原位置刷新的方式
print(len(gt_sentences), len(gt_partition))
track_progress(main, list(zip(gt_sentences, gt_partition)))
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
OOV_Recall = OOV_R / (OOV + 1e-6)
IV_Recall = IV_R / (IV + 1e-6)
print(TP, FN, FP, IV, IV_R, OOV, OOV_R)
print('\n', Precision, Recall, F1, OOV_Recall, IV_Recall)
#并行任务的跟踪进度
# mmcv.track_parallel_progress(func,tasks,nproc)
#刷新位置的进度条方式
# mmcv.track_iter_progress(tasks)
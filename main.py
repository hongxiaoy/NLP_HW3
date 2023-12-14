from functools import partial
import threading
from mmcv import track_progress
from tqdm import tqdm
from tqdm.contrib import tzip

from data import load_gt_partition, load_gt_sentences, load_vocab

from fmm import partition as f_mm
from bmm import partition as b_mm
from bimm import partition as bi_mm

gt_partition = load_gt_partition()
gt_sentences = load_gt_sentences()
vocab = load_vocab()


def calc_metrics(sent, gt, pred, v):
    """
        sent: sentence.
        gt: gt partition.
        pred: pred partition.
        v: vocabulary.
        method: forward, backward, or bi-directional.
    """
    # 将分词结果转换为区间
    gt_number = []
    start = 0
    for w in gt:
        end = start + len(w)
        gt_number.append((start, end))
        start = end
    gt_number = set(gt_number)
    # 将分词结果转换为区间
    pred_number = []
    start = 0
    for w in pred:
        end = start + len(w)
        pred_number.append((start, end))
        start = end
    pred_number = set(pred_number)
    
    tp = len(gt_number & pred_number)
    fp = len(pred_number) - tp
    fn = len(gt_number) - tp

    IV = 0
    OOV = 0
    IV_R = 0
    OOV_R = 0

    for (start, end) in gt_number:
        word = sent[start:end]
        if word in v:
            IV += 1
        else:
            OOV += 1
    
    
    for (start, end) in gt_number & pred_number:
        word = sent[start:end]
        if word in v:
            IV_R += 1
        else:
            OOV_R += 1
    
    return tp, fp, fn, IV, IV_R, OOV, OOV_R

F_TP = 0
F_FP = 0
F_FN = 0

B_TP = 0
B_FP = 0
B_FN = 0

BI_TP = 0
BI_FP = 0
BI_FN = 0

def main(sent_part):

    global F_TP, F_FP, F_FN, B_TP, B_FP, B_FN, BI_TP, BI_FP, BI_FN
    
    s = sent_part[0]
    p = sent_part[1]

    pred_f = f_mm(s, vocab)
    pred_b = b_mm(s, vocab)
    pred_bi = bi_mm(s, vocab)

    preds = [pred_f, pred_b, pred_bi]
    methods = ['f', 'b', 'bi']

    for pred, method in zip(preds, methods):
        tp, fp, fn, IV, IV_R, OOV, OOV_R = calc_metrics(s, p, pred, vocab)
        if method == 'f':
            F_TP += tp
            F_FP += fp
            F_FN += fn
        elif method == 'b':
            B_TP += tp
            B_FP += fp
            B_FN += fn
        elif method == 'bi':
            BI_TP += tp
            BI_FP += fp
            BI_FN += fn
    print(F_TP, F_FP, F_FN, B_TP, B_FP, B_FN, BI_TP, BI_FP, BI_FN)
    print(F_TP/(F_TP+F_FP), F_TP/(F_TP+F_FN))
    print(B_TP/(B_TP+B_FP), B_TP/(B_TP+B_FN))
    print(BI_TP/(BI_TP+BI_FP), BI_TP/(BI_TP+BI_FN))

    print("="*50)


#对一系列项目和任务跟踪进度，进度条原位置刷新的方式
print(len(gt_sentences), len(gt_partition))
# track_progress(main, list(zip(gt_sentences, gt_partition)))
for inp in tqdm(list(zip(gt_sentences, gt_partition))):
    main(inp)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 = 2 * Precision * Recall / (Precision + Recall)
# OOV_Recall = OOV_R / (OOV + 1e-6)
# IV_Recall = IV_R / (IV + 1e-6)
# print('\n', Precision, Recall, F1, OOV_Recall, IV_Recall)
#并行任务的跟踪进度
# mmcv.track_parallel_progress(func,tasks,nproc)
#刷新位置的进度条方式
# mmcv.track_iter_progress(tasks)
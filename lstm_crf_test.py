from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

from numpy.random import permutation
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torchcrf import CRF

np.random.seed(21)
writer = SummaryWriter() 

from data import prepare_nn_gt, load_gt_sentences


from lstm_crf_myself import PartitionModel, PartitionDataset, my_collate

    

def main():
    train_sentences = load_gt_sentences()
    sentences = [
        "这个仅仅是一个小测试",
        "这仅仅是一个小测试",
        "李小福是创新办主任也是云计算方面的专家",
        "实现祖国的完全统一是海内外全体中国人的共同心愿",
        "南京市长江大桥",
        "中文分词在中文信息处理中是最最基础的无论机器翻译亦或信息检索还是其他相关应用如果涉及中文都离不开中文分词因此中文分词具有极高的地位",
        "蔡英文和特朗普通话",
        "研究生命的起源",
        "他从马上下来",
        "老人家身体不错",
        "老人家中很干净",
        "这的确定不下来",
        "乒乓球拍卖完了",
        "香港中文大学将来合肥一中进行招生宣传今年在皖招8人万家热线安徽第一门户",
        "在伦敦奥运会上将可能有一位沙特阿拉伯的女子",
        "美军中将竟公然说",
        "北京大学生喝进口红酒",
        "在北京大学生活区喝进口红酒",
        "将信息技术应用于教学实践",
        "天真的你",
        "我们中出了一个叛徒",
        train_sentences[2],
    ]

    train_dataset = PartitionDataset(True)
    test_dataset = PartitionDataset(False)
    train_loader = DataLoader(train_dataset, 128, True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, 256, False, collate_fn=my_collate)

    model = PartitionModel(train_dataset)

    parameter = torch.load('lstm_crf_partition_best_at_epoch_16.pth', map_location='cpu')
    model.load_state_dict(parameter)

    print(model.crf.start_transitions)
    print(model.crf.transitions)
    print(model.crf.end_transitions)

    if torch.cuda.is_available():
        model = model.to('cuda:0')

    model.eval()

    max_length = train_dataset.max_len

    padded_sentences = []
    for sentence in sentences:
        sentence = [s for s in sentence]
        if max_length - len(sentence) > 0:
            sentence = sentence + ['<PAD>'] * (max_length - len(sentence))
        else:
            sentence = sentence[:max_length]
        for i in range(len(sentence)):
            try:
                sentence[i] = train_dataset.char_to_idx[sentence[i]]
            except:
                # train_dataset.idx_to_char.append(sentence[i])
                # train_dataset.char_to_idx[sentence[i]] = len(train_dataset.idx_to_char) - 1
                sentence[i] = train_dataset.char_to_idx['<UNK>']
        padded_sentences.append(sentence)
    
    model_input = torch.tensor(padded_sentences)

    if torch.cuda.is_available():
        model_input = model_input.to('cuda:0')
    
    mask = model_input != train_dataset.char_to_idx['<PAD>']
    model_output = model(model_input, mask=mask)
    
    for i in range(len(sentences)):
        print("="*100)
        s = sentences[i]
        sep = model_output[i][:len(s)]
        out_s = ''
        for j in range(len(sep)):
            if sep[j] == 3:
                out_s = out_s + '/' + s[j] + '/'
            elif sep[j] == 2:
                out_s = out_s + s[j] + '/'
            elif sep[j] == 0:
                out_s = out_s + '/' + s[j]
            else:
                out_s = out_s + s[j]

        print(s)
        print(sep)
        print(out_s)


if __name__ == "__main__":
    main()
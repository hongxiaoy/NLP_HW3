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

from data import prepare_nn_gt


class PartitionDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()

        self.idx_to_char = []
        self.char_to_idx = {}

        self.gt_sentences, self.gt_labels = prepare_nn_gt()
        self.choosed = permutation(np.arange(len(self.gt_sentences)))
        n_train = int(0.7 * len(self.gt_sentences))

        max_len = 0
        total_len = 0
        lens = []
        for sentence in self.gt_sentences:
            total_len += len(sentence)
            lens.append(len(sentence))
            if len(sentence) > max_len:
                max_len = len(sentence)
        self.max_len = max_len
        self.mean_len = int(total_len / len(self.gt_sentences))
        self.mean_len = int(np.median(np.array(lens)))
        self.mean_len = 32
        self.max_len = 32

        self.label_map = {
            'B': 0,
            'M': 1,
            'E': 2,
            'S': 3,
            '<PAD>': 4,
        }

        for sentence in self.gt_sentences:  # "我喜欢看电影"
            sentence = [s for s in sentence]  # ['我', '喜', '欢', '看', '电', '影']
            self.idx_to_char.extend(sentence)
        self.idx_to_char.append('<UNK>')
        self.idx_to_char.append('<PAD>')
        self.idx_to_char = list(set(self.idx_to_char))
        self.char_to_idx = {k: i for i, k in enumerate(self.idx_to_char)}
        
        
        if train:
            self.choosed = self.choosed[:n_train]
            self.gt_sentences = np.array(self.gt_sentences, dtype=object)[self.choosed].tolist()
            self.gt_labels = np.array(self.gt_labels, dtype=object)[self.choosed].tolist()

            for i in range(len(self.gt_sentences)):
                self.gt_sentences[i] = [s for s in self.gt_sentences[i]]
                if self.mean_len - len(self.gt_sentences[i]) > 0:
                    self.gt_sentences[i] += ['<PAD>'] * (self.mean_len - len(self.gt_sentences[i]))
                else:
                    self.gt_sentences[i] = self.gt_sentences[i][:self.mean_len]
                for j in range(len(self.gt_sentences[i])):
                    self.gt_sentences[i][j] = self.char_to_idx[self.gt_sentences[i][j]]
            
            for i in range(len(self.gt_labels)):
                if self.mean_len - len(self.gt_labels[i]) > 0:
                    self.gt_labels[i] += ['<PAD>'] * (self.mean_len - len(self.gt_labels[i]))
                else:
                    self.gt_labels[i] = self.gt_labels[i][:self.mean_len]
                for j in range(len(self.gt_labels[i])):
                    self.gt_labels[i][j] = self.label_map[self.gt_labels[i][j]]

            assert len(self.gt_sentences) == len(self.gt_labels)       

        else:
            self.choosed = self.choosed[n_train:]
            self.gt_sentences = np.array(self.gt_sentences, dtype=object)[self.choosed].tolist()
            self.gt_labels = np.array(self.gt_labels, dtype=object)[self.choosed].tolist()

            for i in range(len(self.gt_sentences)):
                self.gt_sentences[i] = [s for s in self.gt_sentences[i]]
                if self.mean_len - len(self.gt_sentences[i]) > 0:
                    self.gt_sentences[i] += ['<PAD>'] * (self.mean_len - len(self.gt_sentences[i]))
                else:
                    self.gt_sentences[i] = self.gt_sentences[i][:self.mean_len]
                for j in range(len(self.gt_sentences[i])):
                    self.gt_sentences[i][j] = self.char_to_idx[self.gt_sentences[i][j]]
            
            for i in range(len(self.gt_labels)):
                if self.mean_len - len(self.gt_labels[i]) > 0:
                    self.gt_labels[i] += ['<PAD>'] * (self.mean_len - len(self.gt_labels[i]))
                else:
                    self.gt_labels[i] = self.gt_labels[i][:self.mean_len]
                for j in range(len(self.gt_labels[i])):
                    self.gt_labels[i][j] = self.label_map[self.gt_labels[i][j]]

            assert len(self.gt_sentences) == len(self.gt_labels)
    
    def __len__(self):
        return len(self.gt_sentences)
    
    def __getitem__(self, index):
        gt_sentence = self.gt_sentences[index]
        gt_label = self.gt_labels[index]
        return (gt_sentence, gt_label)


class PartitionModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        n_chars = len(dataset.idx_to_char)

        self.embedding = nn.Embedding(n_chars, 256)
        self.lstm = nn.LSTM(256, 512, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512 * 2, 5)
        self.crf = CRF(5, batch_first=True)  
    
    def forward(self, sentences, tags=None, mask=None):
        lstm_input = self.embedding(sentences)  # (bs, seq_len, embed_dim)
        lstm_output = self.lstm(lstm_input)[0]  # (bs, seq_len, 2 * embed_dim)
        logits = self.fc(lstm_output)  # (bs, seq_len, 5)

        if tags is None:  
            preds = self.crf.decode(logits, mask=mask)  # 解码预测标签  
            return preds  
        else:  
            loss = -self.crf(logits, tags, reduction='mean', mask=mask)  # 计算损失  
            return loss
        
        # Softmax version
        # output = F.softmax(output, dim=-1)



def my_collate(batch_data):
    ret_sentences = []
    ret_labels = []
    for data in batch_data:
        ret_sentences.append(data[0])
        ret_labels.append(data[1])
    ret_sentences = torch.tensor(ret_sentences, dtype=torch.long)
    ret_labels = torch.tensor(ret_labels, dtype=torch.long)
    return ret_sentences, ret_labels
    

def main():
    sentences = [
        "这个仅仅是一个小测试",
        "这仅仅是一个小测试",
        "李小福是创新办主任也是云计算方面的专家",
        "实现祖国的完全统一是海内外全体中国人的共同心愿",
        "南京市长江大桥",
        "中文分词在中文信息处理中是最最基础的，无论机器翻译亦或信息检索还是其他相关应用，如果涉及中文，都离不开中文分词，因此中文分词具有极高的地位。",
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
    ]

    train_dataset = PartitionDataset(True)
    test_dataset = PartitionDataset(False)
    train_loader = DataLoader(train_dataset, 128, True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, 256, False, collate_fn=my_collate)

    model = PartitionModel(train_dataset)

    parameter = torch.load('lstm_crf_partition_best_at_epoch_8.pth', map_location='cpu')
    print(parameter)
    model.load_state_dict(parameter)

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
        sep = model_output[i]
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
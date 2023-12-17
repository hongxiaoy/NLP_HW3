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


from gensim.models import KeyedVectors
char_model = KeyedVectors.load_word2vec_format('wiki.zh.text.char.vector', binary=False)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, torch.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


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
        print(self.mean_len)

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

        self.used_embedding = {}
        for i, ch in enumerate(self.idx_to_char):
            try:
                temp = char_model.key_to_index[ch]  # get index in pretrained embedding
                embed = char_model.vectors[temp]  # get pretrained embedding
                self.used_embedding[i] = embed  # add to mapping from new index to embedding
            except:
                self.used_embedding[i] = None
        
        
        if train:
            self.choosed = self.choosed[:n_train]
            # self.choosed = self.choosed[:]
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

        self.embedding = nn.Embedding(n_chars, 100)
        embedding = torch.randn((n_chars, 100))
        pretrained_embedding = dataset.used_embedding
        for k, v in pretrained_embedding.items():
            if v is not None:
                embedding[int(k)] = torch.tensor(v)
        self.embedding.weight = nn.Parameter(embedding)
        self.lstm = nn.LSTM(100, 512, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512 * 2, 5)

        self.crf = CRF(5, batch_first=True)

        # self.last_crf_transitions = self.crf.transitions
    
    
    def forward(self, sentences, tags=None, mask=None):
        lstm_input = self.embedding(sentences)  # (bs, seq_len, embed_dim)
        lstm_output = self.lstm(lstm_input)[0]  # (bs, seq_len, 2 * embed_dim)
        logits = self.fc(lstm_output)  # (bs, seq_len, 5)
        logits = F.sigmoid(logits)
        # logits = logits[:, :, :4]

        if tags is None:  
            preds = self.crf.decode(logits)  #, mask=mask)  # 解码预测标签  
            return preds  
        else:  
            loss = -self.crf(logits, tags, reduction='mean')  #, mask=mask)  # 计算损失  
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
    total_epoch = 50

    train_dataset = PartitionDataset(True)
    test_dataset = PartitionDataset(False)
    train_loader = DataLoader(train_dataset, 128, True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, 256, False, collate_fn=my_collate)

    model = PartitionModel(train_dataset)
    # print(model.crf.start_transitions)
    # print(model.crf.transitions)
    for para in model.crf.named_parameters():
        if 'transitions' in para[0]:
            print(para)
    # criterion = nn.CrossEntropyLoss(ignore_index=4)
    # optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    if torch.cuda.is_available():
        model = model.to('cuda:0')

    best_acc = 0
    for epoch in trange(total_epoch):
        epoch_loss = []
        for i, batch_data in enumerate(tqdm(train_loader)):
            sentences, labels = batch_data
            if torch.cuda.is_available():
                sentences, labels = sentences.to('cuda:0'), labels.to('cuda:0')
            mask = labels != 4
            mask = sentences != train_dataset.char_to_idx['<PAD>']
            # loss = model.neg_log_likelihood(sentences, tags=labels, mask=mask)
            loss = model(sentences, labels)  #, mask)
            
            # loss = criterion(preds, labels)
            epoch_loss.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for para in model.crf.named_parameters():
            #     if 'transitions' in para[0]:
            #         print(para)

            print(f'\nEpoch: {epoch+1}, Batch: {i+1}, Loss: {loss.cpu().detach().numpy()}')
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
        scheduler.step()
        print(f"\nEpoch: {epoch+1}, Loss: {np.array(epoch_loss).mean()}")

        model.eval()
        sentences_val = []
        labels_val = []
        preds_val = []
        for i, batch_data in enumerate(tqdm(test_loader)):
            sentences, labels = batch_data

            if torch.cuda.is_available():
                sentences, labels = sentences.to('cuda:0'), labels.to('cuda:0')

            sentences_val.append(sentences)
            labels_val.append(labels)

            if torch.cuda.is_available():
                sentences, labels = sentences.to('cuda:0'), labels.to('cuda:0')
            
            mask = sentences != train_dataset.char_to_idx['<PAD>']
            preds = model(sentences)  #, mask=mask)
            # preds = torch.argmax(preds, dim=-1)
            for i in range(len(preds)):
                if len(preds[i]) < 32:
                    preds[i] += [4] * (32 - len(preds[i]))
                else:
                    preds[i] = preds[i][:32]
            preds_val.extend(preds)
        
        sentences_val = torch.vstack(sentences_val)
        labels_val = torch.vstack(labels_val)
        preds_val = torch.tensor(preds_val).to(sentences_val.device)

        labels_mask = torch.where(labels_val != 4)
        labels_val = labels_val[labels_mask]
        preds_val = preds_val[labels_mask]
        TP = torch.sum(labels_val == preds_val)
        TP_and_FN = torch.sum(labels_val == labels_val)
        TP_and_FP = torch.sum(preds_val == preds_val)

        acc = TP / TP_and_FP
        if acc > best_acc:
            torch.save(model.state_dict(), f'lstm_crf_partition_best_at_epoch_{epoch+1}.pth')
            best_acc = acc

        writer.add_scalar('validation Precision', TP/TP_and_FP, epoch)
        writer.add_scalar('validation Recall', TP/TP_and_FN, epoch)
        
        print(f"\nEpoch (Val): {epoch+1}, ACC: {acc}")
            
        model.train()
  
    # 关闭SummaryWriter  
    writer.close()


if __name__ == "__main__":
    main()
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
        for sentence in self.gt_sentences:
            if len(sentence) > max_len:
                max_len = len(sentence)
        self.max_len = max_len

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
        self.idx_to_char.append('<PAD>')
        self.idx_to_char = list(set(self.idx_to_char))
        self.char_to_idx = {k: i for i, k in enumerate(self.idx_to_char)}
        
        
        if train:
            self.choosed = self.choosed[:n_train]
            self.gt_sentences = np.array(self.gt_sentences, dtype=object)[self.choosed].tolist()
            self.gt_labels = np.array(self.gt_labels, dtype=object)[self.choosed].tolist()

            for i in range(len(self.gt_sentences)):
                self.gt_sentences[i] = [s for s in self.gt_sentences[i]]
                self.gt_sentences[i] += ['<PAD>'] * (self.max_len - len(self.gt_sentences[i]))
                for j in range(len(self.gt_sentences[i])):
                    self.gt_sentences[i][j] = self.char_to_idx[self.gt_sentences[i][j]]
            
            for i in range(len(self.gt_labels)):
                self.gt_labels[i] += ['<PAD>'] * (self.max_len - len(self.gt_labels[i]))
                for j in range(len(self.gt_labels[i])):
                    self.gt_labels[i][j] = self.label_map[self.gt_labels[i][j]]

            assert len(self.gt_sentences) == len(self.gt_labels)       

        else:
            self.choosed = self.choosed[n_train:]
            self.gt_sentences = np.array(self.gt_sentences, dtype=object)[self.choosed].tolist()
            self.gt_labels = np.array(self.gt_labels, dtype=object)[self.choosed].tolist()

            for i in range(len(self.gt_sentences)):
                self.gt_sentences[i] = [s for s in self.gt_sentences[i]]
                self.gt_sentences[i] += ['<PAD>'] * (self.max_len - len(self.gt_sentences[i]))
                for j in range(len(self.gt_sentences[i])):
                    self.gt_sentences[i][j] = self.char_to_idx[self.gt_sentences[i][j]]
            
            for i in range(len(self.gt_labels)):
                self.gt_labels[i] += ['<PAD>'] * (self.max_len - len(self.gt_labels[i]))
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
    
    def forward(self, sentences, tags=None):
        lstm_input = self.embedding(sentences)  # (bs, seq_len, embed_dim)
        lstm_output = self.lstm(lstm_input)[0]  # (bs, seq_len, 2 * embed_dim)
        logits = self.fc(lstm_output)  # (bs, seq_len, 5)

        if tags is None:  
            preds = self.crf.decode(logits)  # 解码预测标签  
            return preds  
        else:  
            loss = -self.crf(logits, tags, reduction='mean')  # 计算损失  
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
    total_epoch = 25

    train_dataset = PartitionDataset(True)
    test_dataset = PartitionDataset(False)
    train_loader = DataLoader(train_dataset, 128, True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, 256, False, collate_fn=my_collate)

    model = PartitionModel(train_dataset)
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
            loss = model(sentences, tags=labels)
            
            # loss = criterion(preds, labels)
            epoch_loss.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
            
            preds = model(sentences)
            # preds = torch.argmax(preds, dim=-1)
            preds_val.extend(preds)
        
        sentences_val = torch.vstack(sentences_val)
        labels_val = torch.vstack(labels_val)
        preds_val = torch.tensor(preds_val).to(sentences_val.device)

        labels_mask = torch.where(labels != 4)
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
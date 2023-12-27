import pickle
import torch.optim as optim
import torch
from tqdm import trange
from tqdm.contrib import tzip

from model import BiLSTM_CRF


def get_words(x, y, id2word, id2tag):
    """转换为中文句子
    
    Args:
        x: 一个句子
        y: 一个预测或标签
    """
    single_word = []  # 存储一个词
    sentence = []

    for j in range(len(x)):  # 遍历每一个字符
        if id2tag[y[j]] == 'B':  # 如果预测的结果是 B
            single_word.append(id2word[x[j]])
        elif id2tag[y[j]] == 'M' and len(single_word) != 0:
            single_word.append(id2word[x[j]])
        elif id2tag[y[j]] == 'E' and len(single_word) != 0:
            single_word.append(id2word[x[j]])
            sentence.append(''.join(single_word))  # 词结束，添加词汇，清空词
            single_word = []
        elif id2tag[y[j]]=='S':
            single_word = [id2word[x[j]]]  # 单字构成词
            sentence.append(''.join(single_word))  # 返回单字词
            single_word = []  # 清空词
        else:
            single_word = []
    
    return '/'.join(sentence), sentence


def train():
    # 加载预处理语料库
    with open('data/preprocessed_corpus.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
    
    # 增加新的 token 标记句子的开始和结尾，
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag2id[START_TAG] = len(tag2id)
    tag2id[STOP_TAG] = len(tag2id)

    # 训练的超参数设置
    EMBEDDING_DIM = 100  # 字嵌入维度
    HIDDEN_DIM = 200  # LSTM 隐藏层神经元个数
    EPOCHS = 1  # 训练轮数
    LR = 0.005  # 学习率
    
    TRAIN = False

    # 构建模型
    model = BiLSTM_CRF(len(word2id), tag2id, EMBEDDING_DIM, HIDDEN_DIM)
    # 构建优化器
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)
    # 设置运算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== INFO === Using device: {device}")
    # 将模型送到指定设备上
    model = model.to(device)
    
    if TRAIN:

        # 训练过程
        for epoch in trange(EPOCHS):
            index = 0
            for sentence, tags in tzip(x_train, y_train):  # 获取一个训练数据和训练标签
                index += 1  # mini-batch 索引

                # 梯度清零
                model.zero_grad()
                # 数据转换为张量
                sentence = torch.tensor(sentence, dtype=torch.long).to(device)
                # 标签转换为张量
                tags = torch.tensor(tags, dtype=torch.long).to(device)
                # 计算损失
                loss = model(sentence, tags)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()

                # 打印日志
                if index % 10000 == 0:
                    print(f"Epoch {epoch} Index {index}")
        
        path_name = "model" + str(epoch) + ".pkl"
        torch.save(model, path_name)
        print(f"model has been saved in {path_name}")
    
    else:
        model = torch.load('model0.pkl')
    
    # 验证过程
    entityres = []
    entityall = []
    for sentence, tags in tzip(x_test, y_test):
        sentence = torch.tensor(sentence, dtype=torch.long).to(device)
        score, predict = model.test(sentence)  # 预测测试集中的句子
        # 评价结果
        result, words = get_words(sentence, predict, id2word, id2tag)
        result_gt, words_gt = get_words(sentence, tags, id2word, id2tag)
        
        entityres.extend(words)
        entityall.extend(words_gt)
        
        # print('='*100)
        # print(result)
        # print(result_gt)
    
    print("=== INFO === postprocessing")
    rightpre = [i for i in entityres if i in entityall]
    print("=== INFO === finished postprocessing")
    
    if len(rightpre) != 0:
        precision = float(len(rightpre)) / len(entityres)
        recall = float(len(rightpre)) / len(entityall)
        print("precision: ", precision)
        print("recall: ", recall)
        print("fscore: ", (2 * precision * recall) / (precision + recall))
    else:
        print("precision: ", 0)
        print("recall: ", 0)
        print("fscore: ", 0)
    
    

train()
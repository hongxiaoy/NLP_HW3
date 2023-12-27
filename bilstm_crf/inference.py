import pickle
import torch.optim as optim
import torch
from tqdm import tqdm, trange
from tqdm.contrib import tzip

from model import BiLSTM_CRF


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
    # "香港中文大学将来合肥一中进行招生宣传今年在皖招8人万家热线安徽第一门户",
    "在伦敦奥运会上将可能有一位沙特阿拉伯的女子",
    "美军中将竟公然说",
    "北京大学生喝进口红酒",
    "在北京大学生活区喝进口红酒",
    "将信息技术应用于教学实践",
    "天真的你",
    "我们中出了一个叛徒",
]


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


def preprocess_sentence(sentence, id2word, word2id):
    line_x = []  # 用于存储这一行的训练数据，用字的索引值表示句子
    for i in range(len(sentence)):
        if sentence[i] == " ":  # 如果为空格
            continue  # 则跳到下一个字符
        if (sentence[i] in id2word):
            line_x.append(word2id[sentence[i]])
        else:
            print(len(id2word))
            id2word.append(sentence[i])
            word2id[sentence[i]] = len(id2word)
            line_x.append(len(id2word))
        
    return line_x

        # lineArr = line.split(" ")  # 句子中的词列表
        # line_y = []  # 用于存储这一个句子对应的 tag 序列
        # for item in lineArr:
        #     line_y.extend(get_list(item))
        # y_data.append(line_y)


def inference():
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
    model = torch.load('model0.pkl')
    # 构建优化器
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)
    # 设置运算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== INFO === Using device: {device}")
    # 将模型送到指定设备上
    model = model.to(device)
    
    # 推理过程
    entityres = []
    entityall = []
    for sentence in sentences:
        original_sent = sentence
        sentence = preprocess_sentence(sentence, id2word, word2id)
        sentence = torch.tensor(sentence, dtype=torch.long).to(device)
        score, predict = model.test(sentence)  # 预测测试集中的句子
        # 评价结果
        result, words = get_words(sentence, predict, id2word, id2tag)
        
        print('='*100)
        print(original_sent)
        print(result)
    
    

inference()
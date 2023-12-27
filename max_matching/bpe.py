import collections

from tqdm import tqdm
from data.data import load_gt_sentences, load_vocab
from fmm import partition as f_mm

  
def merge_words(word_lists):  
    # 统计相邻词对的频率  
    counter = collections.Counter()
    for word_list in word_lists:
        for i in range(len(word_list) - 1):  
            counter[word_list[i], word_list[i+1]] += 1  
    
    # 定义一个集合用于保存出现频率最高的词对  
    most_common = counter.most_common(len(counter))  
  
    # 初始化一个空列表用于保存BPE结果  
    bpe_result = []  
  
    # 执行BPE迭代合并操作  
    while len(most_common) > 0:  
        # 从出现频率最高的词对开始合并  
        pair = most_common.pop(0)
        bpe_result.append(pair[0][0] + pair[0][1])  # 合并两个词成一个子词  
        most_common = [x for x in most_common if x[0] not in {pair[0][0], pair[0][1]}]  # 过滤掉已经合并的词对  
  
    return bpe_result  
  
# 示例用法  
sentences = [['我', '爱', '自然语言处理'], ['他', '每天', '早上', '跑步', '锻炼身体']]  
bpe_results = [merge_words(sentence) for sentence in sentences]  
print(bpe_results)





if __name__ == "__main__":
    vocab = load_vocab()
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
    sentences = load_gt_sentences()
    print(sentences[:5])
    corpus_result = []
    for s in tqdm(sentences[:100]):
        seg_result = f_mm(s, vocab)
        corpus_result.append(seg_result)
    bpe_results = merge_words(corpus_result)
    print(bpe_results)
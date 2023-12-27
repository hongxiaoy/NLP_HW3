from data.data import load_vocab
from fmm import partition as f_mm


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

    for s in sentences:
        print("="*100)
        print(s)
        print(f_mm(s, vocab))
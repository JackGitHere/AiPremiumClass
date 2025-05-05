# 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。

# 词向量 word embedding
import jieba
import fasttext

# 分词 预处理
# with open('hlm_c.txt', 'r', encoding='utf-8') as f:
#     lines = f.read()
    

# with open('hlm_sprase_c.txt', 'w', encoding='utf-8') as f:
#     f.write(' '.join(jieba.cut(lines)))
    


# 训练词向量
model = fasttext.train_unsupervised('hlm_sprase_c.txt', model='skipgram')

print("文档词汇表长度", len(model.words))

# 获取词向量
print(model.get_word_vector("宝玉"))

# 获取相似词
print(model.get_nearest_neighbors("宝玉", k=5))

# 分析词间类比
print(model.get_analogies("宝玉", "黛玉", "宝钗"))
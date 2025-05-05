# 1. 实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）

import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bm25_code import bm25

# 读取数据
def load_data(filename):
    # 图书评论信息集合 {"书名": "评论"}
    book_comments = {}
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for item in reader:
            # print(item)
            book = item['book']
            if book == '': continue
            comment = item['body']
            if comment == None: continue
            comment_words = jieba.lcut(comment)
            
            
            
            # 收集数据
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    return book_comments

# tf-idf | bm25 推荐
def comments_vector_similarity(book_comms, method="bm25"):
    if method == "tfidf":
        # 构建TF-IDF特征矩阵
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])
    if method == "bm25":
        # 构建BM25特征矩阵
        matrix = bm25([' '.join(comms) for comms in book_comms])
    # 计算相似度
    similarity = cosine_similarity(matrix)
    return similarity

if __name__ == "__main__":
    # 加载截停词
    stop_words = [line.strip() for line in open('stopwords.txt', 'r').readlines()]
    
    # 加载图书评论信息
    book_comments = load_data('doubantop.txt')
    
    # 提取书名和评论文本
    book_names = []
    book_comms = []
    for book, comms in book_comments.items():
        book_names.append(book)
        book_comms.append(comms)

    # TF-IDF => 计算相似度
    tfidf_matrix = comments_vector_similarity(book_comms, method="tfidf")
    # BM25 => 计算相似度
    bm25_matrix = comments_vector_similarity(book_comms, method="bm25")
    
    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入要推荐的图书名称：")
    book_idx = book_list.index(book_name)
    
    # 推荐相似图书
    print(f"TF-IDF 推荐相似图书：\n")
    recommend_book_index = np.argsort(-tfidf_matrix[book_idx])[1:11]
    for i in recommend_book_index:
        print(f"书名：{book_list[i]}  相似度：{tfidf_matrix[book_idx][i]:.4f}")
    print()
    
    print(f"BM25 推荐相似图书：\n")
    recommend_book_index = np.argsort(-bm25_matrix[book_idx])[1:11]
    for i in recommend_book_index:
        print(f"书名：{book_list[i]}  相似度：{bm25_matrix[book_idx][i]:.4f}")
        
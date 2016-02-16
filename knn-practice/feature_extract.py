import numpy as np
import re

def extract(data):
    # features = char_len(data)
    # features = word_count(data)
    features = keyword_count(data)
    return np.array(features)

# 本文文字列の長さ
def char_len(data):
    features = [[len(feature)] for feature in data[:,3]]
    return features

# 本文の単語数
def word_count(data):
    features = [[feature.count(" ")] for feature in data[:,3]]
    return features

def keyword_count(data):
    keywords = ["food", "travel", "fun", "innovation", "shopping"]
    features = []
    for feature in data[:,3]:
        counts = []
        for keyword in keywords:
            repatter = re.compile(keyword, re.IGNORECASE)
            count = len(repatter.findall(feature))
            counts.append(count)
        features.append(counts)
    # print(features)
    return features

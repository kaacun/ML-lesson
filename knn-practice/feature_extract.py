import numpy as np

def extract(data):
    features = char_len(data)
    # features = word_count(data)
    return np.array(features)

# 本文文字列の長さ
def char_len(data):
    features = [[len(feature)] for feature in data[:,3]]
    return features

# 本文の単語数
def word_count(data):
    features = [[feature.count(" ")] for feature in data[:,3]]
    return features

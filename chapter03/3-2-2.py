import os
import scipy as sp
import sys

# 類似度の計算
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

# txt読み込み
posts = [open(os.path.join("data/toy/", f)).read() for f in os.listdir("data/toy/")]
from sklearn.feature_extraction.text import CountVectorizer
# ベクトル化の実行
vectorizer = CountVectorizer(min_df = 1)
X_train = vectorizer.fit_transform(posts)
# 文書と単語数の取得
num_samples, num_features = X_train.shape

# 新しい文書のベクトル化
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

dist = dist_raw
best_dist = sys.maxsize
best_i = None

# 文書間の距離を計算し、最も近い文書を探す
for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist(post_vec, new_post_vec)

    print("=== Post %i with dist=%.2f: %s" % (i + 1, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i

print("Best post is %i with dist=%.2f" % (best_i + 1, best_dist))

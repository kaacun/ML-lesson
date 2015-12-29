import os
import scipy as sp
import sys
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 類似度の計算(正規化あり)
def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized

    return sp.linalg.norm(delta.toarray())

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words='english', decode_error='ignore')

# txt読み込み
posts = [open(os.path.join("data/toy/", f)).read() for f in os.listdir("data/toy/")]
# ベクトル化の実行
X_train = vectorizer.fit_transform(posts)
# 文書と単語数の取得
num_samples, num_features = X_train.shape

# 新しい文書のベクトル化
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

dist = dist_norm
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

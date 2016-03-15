from load import load_dataset
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp

# データ読み込み
raw_data, labels = load_dataset('aaj_data1000')
content = raw_data[:,3]
num_clusters = 5

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                    stop_words='english', decode_error='ignore'
                                    )

vectorized = vectorizer.fit_transform(content)

km = KMeans(n_clusters=num_clusters, n_init=1, verbose=1)
clustered = km.fit(vectorized)

# print(km.cluster_centers_[6])
# np.savetxt('test.csv', km.cluster_centers_[0], delimiter=',')

# 全クラスタについて、クラスタ中心との距離を計算
for cluster_number in range(0, num_clusters):
    similar_indices = (km.labels_ == cluster_number).nonzero()[0]
    
    # クラスタ中心との距離を計算
    similar = []
    for i in similar_indices:
        dist = sp.linalg.norm((km.cluster_centers_[cluster_number] - vectorized[i].toarray()))
        similar.append((dist, content[i]))
    
    # クラスタ中心との距離が近い順に並べ替え
    similar = sorted(similar)
    
    # ファイル書き込み
    f = open("similar_" + str(cluster_number) + ".txt","w")
    for i in similar:
        f.write(str(i[0]))
        f.write("   ")
        f.write(i[1])
        f.write("\n")
    f.close()

# 評価指標を計算
from sklearn import metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand Index: %0.3f" %
      metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted Mutual Information: %0.3f" %
      metrics.adjusted_mutual_info_score(labels, km.labels_))
print(("Silhouette Coefficient: %0.3f" %
       metrics.silhouette_score(vectorized, labels, sample_size=1000)))

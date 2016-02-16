from load import load_dataset
from feature_extract import extract
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import numpy as np
from utils import plot_features

# データ読み込み
raw_data, labels = load_dataset('aaj_data1000')
# 特徴抽出
features = extract(raw_data)
plot_features(features)

classifier = KNeighborsClassifier(n_neighbors=6)

means = []
# K分割交差検定
kf = KFold(len(features), n_folds=3, shuffle=True)
for training,testing in kf:
    # 訓練データで学習する
    classifier.fit(features[training], labels[training])
    # テストデータで検証する
    prediction = classifier.predict(features[testing])
    curmean = np.mean(prediction == labels[testing])
    means.append(curmean)
print('Result of cross-validation using KFold: {}'.format(means))

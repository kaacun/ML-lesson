from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
#from pprint import pprint

data = load_iris()
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

#pprint(target)

fig,axes = plt.subplots(2, 3)
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

color_markers = [
        ('r', '>'),
        ('g', 'o'),
        ('b', 'x'),
      ]

for i, (p0, p1) in enumerate(pairs):
    ax = axes.flat[i]

    for t in range(3):
        c,marker = color_markers[t] # 特徴量ことに異なる色と形を設定する
        ax.scatter(features[target == t, p0],
                   features[target == t, p1],
                   marker = marker,
                   c = c) # scatter : 分布図を描画するためのメソッド
        ax.set_xlabel(feature_names[p0])
        ax.set_ylabel(feature_names[p1])
        ax.set_xticks([]) # x軸の目盛
        ax.set_yticks([]) # y軸の目盛

fig.tight_layout()
fig.savefig('figure1.png', dpi=300)

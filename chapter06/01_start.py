# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#
# This script trains multinomial Naive Bayes on the tweet corpus
# to find two different results:
# - How well can we distinguis positive from negative tweets?
# - How well can we detect whether a tweet contains sentiment at all?
#

import time
start_time = time.time()

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit

from utils import plot_pr
from utils import load_sanders_data
from utils import tweak_labels

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB


def create_ngram_model():
    # 3-gramでのTF-IDF
    tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3),
                                   analyzer="word", binary=False)
    clf = MultinomialNB()
    # Pipelineクラスでベクトル化を行う機能と分類器を合わせて使える
    # 通常の分類器と同じようにfitとpredictメソッドを持つ
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    return pipeline


def train_model(clf_factory, X, Y, name="NB ngram", plot=False):
    # 交差検定のデータシャッフル版
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []

    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf = clf_factory()
        # 学習
        clf.fit(X_train, y_train)

        # 平均正解率を計算
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        # 確率推定を求める
        proba = clf.predict_proba(X_test)

        # fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        # 適合率、再現率を求める
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test, proba[:, 1])

        # AUC
        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

    scores_to_sort = pr_scores
    median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

    # グラフ描画
    if plot:
        plot_pr(pr_scores[median], name, "01", precisions[median],
                recalls[median], label=name)

        summary = (np.mean(scores), np.std(scores),
                   np.mean(pr_scores), np.std(pr_scores))
        print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    return np.mean(train_errors), np.mean(test_errors)


def print_incorrect(clf, X, Y):
    Y_hat = clf.predict(X)
    wrong_idx = Y_hat != Y
    X_wrong = X[wrong_idx]
    Y_wrong = Y[wrong_idx]
    Y_hat_wrong = Y_hat[wrong_idx]
    for idx in range(len(X_wrong)):
        print("clf.predict('%s')=%i instead of %i" %
              (X_wrong[idx], Y_hat_wrong[idx], Y_wrong[idx]))


if __name__ == "__main__":
    # tweetデータ読み込み
    # X_orig:tweet内容 Y_orig:ラベル(positive/negative)
    X_orig, Y_orig = load_sanders_data()
    classes = np.unique(Y_orig)
    for c in classes:
        print("#%s: %i" % (c, sum(Y_orig == c)))

    print("== Pos vs. neg ==")
    # positiveかnegativeのもののみ抽出
    pos_neg = np.logical_or(Y_orig == "positive", Y_orig == "negative")
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]
    # positive/negativeを1/0に変換
    Y = tweak_labels(Y, ["positive"])

    train_model(create_ngram_model, X, Y, name="pos vs neg", plot=True)

    print("== Pos/neg vs. irrelevant/neutral ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])
    train_model(create_ngram_model, X, Y, name="sent vs rest", plot=True)

    print("== Pos vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    train_model(create_ngram_model, X, Y, name="pos vs rest", plot=True)

    print("== Neg vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    train_model(create_ngram_model, X, Y, name="neg vs rest", plot=True)

    print("time spent:", time.time() - start_time)

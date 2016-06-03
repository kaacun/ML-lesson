# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#
# This script tries to improve the classifier by cleaning the tweets a bit
#

import time
start_time = time.time()
import re

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from utils import plot_pr
from utils import load_sanders_data
from utils import tweak_labels
from utils import log_false_positives

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
import pickle

phase = "03"

# 顔文字
emo_repl = {
    # positive emoticons
    "&lt;3": " good ",
    ":d": " good ",  # :D in lower case
    ":dd": " good ",  # :DD in lower case
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",

    # negative emoticons:
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":S": " bad ",
    ":-S": " bad ",
}

# 例えば、":dd"より先に":d"で置き換えられないようにするため、
# 置き換えの順番を決める
emo_repl_order = [k for (k_len, k) in reversed(
    sorted([(len(k), k) for k in list(emo_repl.keys())]))]

# 略語
re_repl = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
}


def create_ngram_model(params=None):
    def preprocessor(tweet):
        global emoticons_replaced
        # 文章を小文字に変換
        tweet = tweet.lower()

        # 顔文字の置換
        for k in emo_repl_order:
            tweet = tweet.replace(k, emo_repl[k])
        # 略語の置換
        for r, repl in re_repl.items():
            tweet = re.sub(r, repl, tweet)

        return tweet

    tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor,
                                   analyzer="word")
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline

# 最適パラメータを求める(時間がかかる)
def grid_search_model(clf_factory, X, Y):
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, random_state=0)

    param_grid = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      vect__min_df=[1, 2],
                      vect__stop_words=[None, "english"],
                      vect__smooth_idf=[False, True],
                      vect__use_idf=[False, True],
                      vect__sublinear_tf=[False, True],
                      vect__binary=[False, True],
                      clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],
                      )

    grid_search = GridSearchCV(clf_factory(),
                               param_grid=param_grid,
                               cv=cv,
                               scoring='f1',
                               verbose=10)
    grid_search.fit(X, Y)
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(best_params)
    # 最適パラメータを保存
    with open('data/params/neg_rest.pickle', 'wb') as f:
        pickle.dump(best_params, f)

    return clf

def train_model(clf, X, Y, name="NB ngram", plot=False):
    # create it again for plotting
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []

    clfs = []  # just to later get the median

    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf.fit(X_train, y_train)
        clfs.append(clf)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        proba = clf.predict_proba(X_test)

        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test, proba[:, 1])

        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

    if plot:
        scores_to_sort = pr_scores
        median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

        plot_pr(pr_scores[median], name, phase, precisions[median],
                recalls[median], label=name)

        log_false_positives(clfs[median], X_test, y_test, name)

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


def get_best_model():
    # 02_tuning.pyで求めた最適パラメータを返す
    best_params = dict(tfidf__ngram_range=(1, 2),
                       tfidf__min_df=1,
                       tfidf__stop_words=None,
                       tfidf__smooth_idf=False,
                       tfidf__use_idf=False,
                       tfidf__sublinear_tf=True,
                       tfidf__binary=False,
                       clf__alpha=0.01,
                       )

    best_clf = create_ngram_model(best_params)
    return best_clf

if __name__ == "__main__":
    X_orig, Y_orig = load_sanders_data()
    classes = np.unique(Y_orig)
    for c in classes:
        print("#%s: %i" % (c, sum(Y_orig == c)))

    print("== Pos vs. neg ==")
    pos_neg = np.logical_or(Y_orig == "positive", Y_orig == "negative")
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]
    Y = tweak_labels(Y, ["positive"])
    best_clf = grid_search_model(create_ngram_model, X, Y)
    train_model(best_clf, X, Y, name="pos vs neg", plot=True)
    # train_model(get_best_model(), X, Y, name="pos vs neg", plot=True)

    print("== Pos/neg vs. irrelevant/neutral ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])
    best_clf = grid_search_model(create_ngram_model, X, Y)
    train_model(best_clf, X, Y, name="pos+neg vs rest", plot=True)
    # train_model(get_best_model(), X, Y, name="pos+neg vs rest", plot=True)

    print("== Pos vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    best_clf = grid_search_model(create_ngram_model, X, Y)
    train_model(best_clf, X, Y, name="pos vs rest", plot=True)
    # train_model(get_best_model(), X, Y, name="pos vs rest", plot=True)

    print("== Neg vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    best_clf = grid_search_model(create_ngram_model, X, Y)
    train_model(best_clf, X, Y, name="neg vs neg", plot=True)
    # train_model(get_best_model(), X, Y, name="neg vs rest", plot=True)

    print("time spent:", time.time() - start_time)

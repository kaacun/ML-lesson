from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)
content = ["How to format my hard disk", "Hard disk format problems"]
# ベクトル化の実行
X = vectorizer.fit_transform(content)
# 特徴量の取得
print(vectorizer.get_feature_names())
# カウント結果の表示
print(X.toarray())

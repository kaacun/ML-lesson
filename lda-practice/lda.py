from load import load_dataset
from load import create_corpus
from wordcloud import create_cloud
from gensim import corpora, models, matutils
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pprint

NUM_TOPICS = 100

# 初回のみ実行
# create_corpus()

# load dictionary
dictionary = corpora.Dictionary.load('../data/aaj.dict')
# load corpus
corpus = corpora.MmCorpus('../data/aaj.mm')

# Build the topic model
model = models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=dictionary, alpha=None)

# We first identify the most discussed topic, i.e., the one with the
# highest total weight
topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
weight = topics.sum(1)
max_topic = weight.argmax()

# Get the top 64 words for this topic
# Without the argument, show_topic would return only 10 words
words = model.show_topic(max_topic, 64)

# ワードクラウドを生成
create_cloud('../data/cloud_lda.png', words)

# トピック数分布をプロット
num_topics_used = [len(model[doc]) for doc in corpus]
fig,ax = plt.subplots()
ax.hist(num_topics_used, np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')
fig.tight_layout()
fig.savefig('../data/Figure_04_01.png')

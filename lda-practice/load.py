import numpy as np
from gensim import corpora, models, matutils

def load_dataset(dataset_name):
    data = []
    labels = []
    with open('../data/{0}.tsv'.format(dataset_name)) as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            data.append(tokens[2:])
            labels.append(tokens[1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def create_corpus():
    # Load the data
    raw_data, labels = load_dataset('aaj_data1000')
    documents = raw_data[:,3]
    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]
    # remove words that appear only once
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once]
             for text in texts]
    # create dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.save('../data/aaj.dict') # store the dictionary, for future reference
    dictionary.save_as_text('../data/aaj_text.dict')

    # create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('../data/aaj.mm', corpus) # store to disk, for later use

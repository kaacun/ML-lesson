import numpy as np

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

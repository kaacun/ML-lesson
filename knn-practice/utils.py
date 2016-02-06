import numpy as np
import matplotlib.pyplot as plt

def plot_features(features):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(features, bins=50)
    ax.set_title('histogram')
    ax.set_xlabel('x')
    ax.set_ylabel('freq')

    filename = "output.png"
    plt.savefig(filename)


import numpy as np
import metric_learn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)

X, y = make_classification(n_samples=100, n_classes=3, n_clusters_per_class=2,
                           n_informative=3, class_sep=4., n_features=5,
                           n_redundant=0, shuffle=True,
                           scale=[1, 1, 20, 20, 20])


def plot_tsne(X, y, colormap=plt.cm.Paired):
    plt.figure(figsize=(8, 6))

    # clean the figure
    plt.clf()

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colormap)
    
    plt.xticks(())
    plt.yticks(())

    plt.show()

def LMNN():
    # setting up LMNN
    lmnn = metric_learn.LMNN(k=5, learn_rate=1e-6)

    # fit the data!
    lmnn.fit(X, y)

    # transform our input space
    X_lmnn = lmnn.transform(X)
    plot_tsne(X_lmnn, y)

# def ITML():
#     itml = metric_learn.ITML_Supervised()
#     X_itml = itml.fit_transform(X, y)

#     plot_tsne(X_itml, y)

# def MMC():
#     mmc = metric_learn.MMC_Supervised()
#     X_mmc = mmc.fit_transform(X, y)

#     plot_tsne(X_mmc, y)

def NCA():
    nca = metric_learn.NCA(max_iter=1000)
    X_nca = nca.fit_transform(X, y)

    plot_tsne(X_nca, y)

def LFDA():
    lfda = metric_learn.LFDA(k=2, n_components=2)
    X_lfda = lfda.fit_transform(X, y)

    plot_tsne(X_lfda, y)


plot_tsne(X,y)

LMNN()

# NCA()

# LFDA()
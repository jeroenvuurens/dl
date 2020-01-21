import torch
import os
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
from pathlib2 import Path
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import sys

def getsizeof(o, ids=set()):
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = sys.getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

def to_numpy(arr):
    if type(arr) is torch.Tensor:
        if arr.device.type == 'cuda':
            return arr.data.cpu().numpy()
        else:
            return arr.data.numpy()
    return arr

def plot_histories(metric, *history, **kwargs):
    plt.figure(**kwargs)
    for i, t in enumerate(history):
        h = t.history
        x = [ epoch['epoch'] for epoch in h.epochs['train'] ]
        plt.plot(x, h.train(metric), label=f'train_{i}')
        plt.plot(x, h.valid(metric), label=f'valid_{i}')
    plt.ylabel(metric.__name__)
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

def create_path(p, mode=0o777):
    path = Path(p)
    os.makedirs(path, mode, exist_ok=True)
    return path

def scatter(x, colors):
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []

    for i in range(num_classes):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    #return f, ax, sc, txts

def to_numpy1(a):
    try:
        a = a.detach()
    except: pass
    try:
        a = a.numpy()
    except: pass
    return a

def draw_regression(x, y_true, y_pred):
    f = plt.figure(figsize=(8, 8))
    x, y_true, y_pred = [to_numpy(a) for a in (x, y_true, y_pred)]
    plt.scatter(x, y_true)
    indices = np.argsort(x)
    plt.plot(x[indices], y_pred[indices])

def line_predict(x, y_true, y_pred):
    draw_regression(x, y_true, y_pred)

def scatter(x, y):
    f = plt.figure(figsize=(8, 8))
    x, y = [to_numpy(a) for a in (x, y)]
    plt.scatter(x, y)

def plot_tsne(X, y, random_state=0):
    t = TSNE(random_state=random_state).fit_transform(X)
    scatter(t, y)



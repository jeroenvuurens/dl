from __future__ import print_function, with_statement, division
import torch
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from .train_metrics import loss, name_metrics
from math import log, exp
import statistics
from functools import partial

def frange(start, end, steps):
    incr = (end - start) / (steps)
    return (start + x * incr for x in range(steps))

def exprange(start, end, steps, **kwargs):
    return (exp(x) for x in frange(log(start), log(end), steps))

def arange(start, end, steps, **kwargs):
    return np.arange(start, end, steps)

def set_dropouts(dropouts):
     def change(value):
        for d in dropouts:
           d.p = value
     return change

class tuner:
    def __init__(self, trainer, values, param_update, label='parameter', smooth=0.05, diverge=25, **kwargs):
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.label = label
        self.trainer = trainer
        self.values = list(values)
        self.param_update = param_update
        self.range_test(smooth, diverge)

    def reset(self):
        self.trainer.revert('tuner')

    def next_train(self):
        try:
            return next(self.train_iterator)
        except (StopIteration, AttributeError):
            self.train_iterator = iter(self.trainer.train_dl)
            return next(self.train_iterator)

    def range_test( self, smooth, diverge):
        self.x = []
        self.loss = []
        self.sloss = []
        self.min_loss = None
        self.trainer.commit('tuner')

        for i in tqdm(self.values):
            self.x.append(i)
            self.param_update(i)
            X, y = self.next_train()
            if self.trainer.device is not None:
                X, y = X.to(self.trainer.device), y.to(self.trainer.device)

            loss, pred_y = self.trainer.train_batch(X, y)
            loss = self.trainer.validate_loss()
            self.loss.append(loss)

            # Track the best loss and smooth it if smooth_f is specified
            try:
                loss = smooth * loss + (1 - smooth) * self.history["loss"][-1]
            except: pass
            self.sloss.append(loss)

            try:
                self.min_loss = min(self.min_loss, loss)
            except:
                self.min_loss = loss

            

            # Check if the loss has diverged; if it has, stop the test
            if loss > diverge * self.min_loss:
                #print("Stopping early, the loss has diverged")
                break
        self.reset()
        #print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def plot(self, log=True):
        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        loss = self.sloss
        imin = loss.index(self.min_loss)
        median = statistics.median(loss[:imin+1])
        self.max_loss = self.min_loss + (median - self.min_loss) * 3

        skip_end = next((x[0]+1 for x in enumerate(loss) if x[0] > imin and x[1] > self.max_loss), len(loss))
        x = self.x[:skip_end]
        loss = loss[:skip_end]

        # Plot loss as a function of the learning rate
        plt.plot( x, loss)
        plt.ylim( self.min_loss, self.max_loss )
        if log:
            plt.xscale("log")
        plt.xlabel(self.label)
        plt.ylabel("Loss")
        plt.show()


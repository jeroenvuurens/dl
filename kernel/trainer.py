import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import timeit
import sys
import copy
import numpy as np
import math
from tqdm import tqdm_notebook as tqdm
from .train_diagnostics import *
from .train_metrics import *
from .train_history import *
from .jcollections import *
from .optimizers import *
from .transfer import *
from .helper import *
from functools import partial
import os
try:
    GPU = int(os.environ['GPU'])
    GPU = 0
except:
    GPU = -1

def last_container(last):
    try:
        children = list(last.children())
    except: return None
    for c in children[::-1]:
        l = last_container(c)
        if l is not None:
            return l
        try:
            if c.out_features > 0:
                return c
        except: pass

#def last_container(last):
#    children = list(last.children())
#    l = []
#    while len(children) > 0:
#        l.append(children[-1])
#        last = children[-1]
#        children = list(last.children())
#
#    return l[-1]

class ordered_dl:
    def __init__(self, dl):
        self.dl = dl

    def __enter__(self):
        self.oldsampler = self.dl.batch_sampler.sampler
        self.newsampler = torch.utils.data.sampler.SequentialSampler(self.oldsampler.data_source)
        self.dl.batch_sampler.sampler = self.newsampler
        return self.dl

    def __exit__(self, exc_type, exc_value, tb):
        self.dl.batch_sampler.sampler = self.oldsampler
        if exc_type is not None:
            return False

class trainer:
    def __init__(self, model, data, report_frequency=1, report_phases=['train','valid'], metrics = [loss, acc], modules=[], optimizer=Adam, loss=cross_entropy_loss, out_features=None, random_state=None, log=True, cycle_epochs=1.0, gpu=GPU, device=None, **kwargs):
        self.report_frequency = report_frequency
        self.report_phases = report_phases
        self.metrics = metrics
        self.modules = modules
        self.loss = loss
        self.random_state = random_state
        self.log = log
        self.cycle_epochs = cycle_epochs
        self.data = data
        self.set_device(gpu, device)
        self._model = model
        try:
            self.post_forward = model.predict
        except: pass
        if out_features is not None:
            self._out_features = out_features
        self.optimizerclass = optimizer
        if self.random_state is not None:
            torch.backends.cudnn.deterministic=True
            torch.manual_seed(self.random_state)
        self.history
        self._commit = {}
        self.epochid = 0
        self.cpu = False

    def __repr__(self):
        return 'Trainer( ' + self.model + ')'

    def set_device(self, gpu, device):
        if device is not None:
            self._device = device
        else:
            if gpu is None:
                self._device = torch.device('cpu')
            elif gpu > -1:
                self._device = torch.device(f'cuda:{gpu}')

    @property
    def device(self):
        try:
            return torch.device('cpu') if self.cpu else self._device
        except:
            return torch.device('cpu')

    @property
    def optimizer(self):
        try:
            return self._optimizer
        except:
            self._optimizer = self.optimizerclass(self.model.parameters())
            return self._optimizer

    def change_lr(self, lr):
        try:
            if self.lr == lr:
                return
        except: pass
        self.lr = lr
        if (type(lr) is list or type(lr) is tuple) and len(lr) == 2:
            lr, *_ = lr
            self.scheduler = cyclical_scheduler(self)
        else:
            self.scheduler = uniform_scheduler(self)
        self.set_lr(lr)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def out_features(self):
        try:
            return self._out_features
        except: pass
        try:
            self._out_features = last_container(self.model).out_features
            return self._out_features
        except:
            print('cannot infer out_features from the model, please specify it in the constructor of the trainer')
            raise

    @property
    def in_features(self):
        first = next(iter(self._model.modules()))
        while type(first) is nn.Sequential:
            first = next(iter(first.modules()))
        return first.in_features

    @property
    def history(self):
        try:
            return self._history
        except:
            self.reset_history()
            return self._history

    def reset_history(self):
        try:
            del self._history
        except: pass
        self._history = train_history(self)
    
    @property
    def train_dl(self):
        return self.data.train_dl

    @property
    def valid_dl(self):
        return self.data.valid_dl

    @property
    def valid_ds(self):
        return self.data.valid_ds

    @property
    def train_ds(self):
        return self.data.train_ds

    @property
    def valid_rows(self):
        return self.data.valid_dl.dataset

    @property
    def valid_tensors(self):
        return self.data.valid_dl.dataset.tensors

    @property
    def train_tensors(self):
        return self.data.train_dl.dataset.tensors

    @property
    def train_X(self):
        return self.train_tensors[0]

    @property
    def train_y(self):
        return self.train_tensors[1]

    @property
    def valid_X(self):
        return self.valid_tensors[0]

    @property
    def valid_y(self):
        return self.valid_tensors[1]

    @property
    def train_rows(self):
        return self.data.train_dl.dataset

    @property
    def model(self):
        try:
            if self.device is not self._model.device:
                self._model.device = self.device
                self._model.to(self.device)
                try:
                    del self._optimizer
                except: pass
        except:
            try:
                self._model.device = self.device
                self._model.to(self.device)
                #print('change device')
                try:
                    del self._optimizer
                except: pass
            except: pass
        return self._model

    def parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def predict(self, X):
        self.model.eval()
        return self.post_forward(self.model(X.to(self.device)))

    def post_forward(self, y):
        return y

    def list_commits(self):
        return self._commit.keys()

    def commit(self, label):
        "save the model and optimzer state, allowing to revert to a previous state"
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self._commit[label] = (model_state, optimizer_state)

    def revert(self, label):
        "revert the model and optimizer to a previously commited state, deletes the commit point"
        if label in self._commit:
            model_state, optimizer_state = self._commit.pop(label)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
        else:
            print('commit point {label} not found')
    
    def checkout(self, label):
        "switches the model and optimizer to a previously commited state, keeps the commit point"
        if label in self._commit:
            model_state, optimizer_state = self._commit[label]
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)  
        else:
            print('commit point {label} not found')

    def purge(self, label):
        "switches the model and optimizer to a previously commited state, keeps only the commit point"
        if label in self._commit:
            self.checkout(label)
            self._commit = { l:s for l, s in self._commit.items() if l == label }
        else:
            print('commit point {label} not found')

    def train_correct(self):
        with ordered_dl(self.train_dl) as dl:
            for i, (x, y) in enumerate(dl):
                y_pred = self.predict(x)
                for ii, yy in zip(i, y == y_pred):
                    print(ii, yy)

    def validate_loss(self):
        with torch.set_grad_enabled(False):
            xys = [ (X.to(self.device), y.to(self.device)) for X, y in self.valid_dl ]
            losses = [ (self.loss_xy(X, y)[0].item(), len(y)) for X, y in xys ]
            sums = [ sum(x) for x in zip(*losses) ]
            return sums[0] / sums[1]

    def validate(self, pbar=None):
        epoch = self.history.create_epoch('valid')
        epoch.report = True
        with torch.set_grad_enabled(False):
            epoch.before_epoch()
            for X, y in self.valid_dl:
                if self.device is not None:
                    X, y = X.to(self.device), y.to(self.device)
                epoch.before_batch(X, y)
                loss, y_pred = self.loss_xy(X, y)
                epoch['loss'] += loss.item() * len(y)
                epoch['n'] += len(y)
                epoch.after_batch( X, y, y_pred, loss )
                if pbar is not None:
                    pbar.update(self.data.batch_size)
            epoch['loss'] /= epoch['n']
            epoch.after_epoch()
        return epoch

    def loss_xy(self, X, y):
        y_pred = self.model(X)
        return self.loss(y_pred, y), self.post_forward(y_pred)

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        loss, y_pred = self.loss_xy(X, y)
        loss.backward()
        self.optimizer.step()
        return loss, y_pred
    
    def train(self, epochs=1, lr=1e-3, report_frequency=None, save=None, gpu=GPU, device=None):
        try:
            del self._optimizer
        except: pass
        #self.cpu=False
        if report_frequency is None:
            report_frequency = self.report_frequency
        self.set_device(gpu, device)
        self.change_lr(lr)
        model = self.model
        model.train()
        reports = math.ceil(epochs / report_frequency)
        maxepoch = self.epochid + epochs
        batches = len(self.train_dl) * self.data.batch_size * epochs + len(self.valid_dl) * self.data.batch_size * reports
        pbar = tqdm(range(batches), desc='Total')
        for i in range(epochs):
            self.epochid += 1
            epoch = self.history.create_epoch('train')
            if self.log and (((i + 1) % report_frequency) == 0 or i == epochs - 1):
                epoch.report = True
            epoch.before_epoch()
            for X, y in self.train_dl:
                if self.device is not None:
                    X, y = X.to(self.device), y.to(self.device)
                epoch.before_batch( X, y )
                self.scheduler.step()
                loss, y_pred = self.train_batch(X, y)
                epoch['loss'] += loss.item() * len(y)
                epoch['n'] += len(y)
                epoch.after_batch( X, y, y_pred, loss)
                pbar.update(self.data.batch_size)
            epoch['loss'] /= epoch['n']
            epoch.after_epoch()
            if epoch.report:
                vepoch = self.validate(pbar = pbar)
                self.history.register_epoch(epoch)
                self.history.register_epoch(vepoch)
                print(f'{self.epochid} {epoch.time():.2f}s {epoch} {vepoch}')
                if save is not None:
                    self.commit(f'{save}-{self.epochid}')
    
    def lr_find(self, start=1e-6, end=10, steps=100, smooth=0.05, device=None, gpu=GPU, **kwargs):
        self.set_device(gpu, device)
        #self.cpu = False
        self._lr_find = tuner(self, exprange(start, end, steps), self.set_lr, label='lr', smooth=smooth)
        #self.set_device(None, None)
        #self.cpu = True
        #_ = self.model.device
        return self._lr_find.plot(**kwargs)

    def plot(self, *metric, **kwargs):
        self.history.plot(*metric, **kwargs)
        

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

class databunch:
    def __init__(self, train_dl, valid_dl, test_dl=None, device=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        if device:
            self._device = device

    @property
    def device(self):
        try:
            return self._device
        except:
            try:
                self._device = next(iter(self.train_dl)).device
            except:
                self._device = next(iter(self.train_dl))[0].device
            return self._device

class trainer:
    def __init__(self, model, db=None, train_dl=None, valid_dl=None, report_frequency=1, report_phases=['train','valid'], metrics = [loss, acc], modules=[], optimizer=AdamW, optimizerparams=dict(), loss=cross_entropy_loss, out_features=None, random_state=None, log=True, cycle_epochs=1.0, scheduler='onecycle', weight_decay=None, momentum=None, device=None, gpu=None, **kwargs):
        self.report_frequency = report_frequency
        self.report_phases = report_phases
        self.metrics = metrics
        self.modules = modules
        self.loss = loss
        self.random_state = random_state
        self.log = log
        self.cycle_epochs = cycle_epochs
        if gpu is not None:
            if gpu == -1:
                device = torch.device('cpu')
            else:
                device = torch.device(f'cuda:{gpu}')
        self.set_data(db=db, train_dl=train_dl, valid_dl=valid_dl, device=device)
        self._model = model
        try:
            self.post_forward = model.predict
        except: pass
        if out_features is not None:
            self._out_features = out_features
        self._optimizerclass = optimizer
        self._optimizerparams = optimizerparams
        self.schedulertype = scheduler
        if self.random_state is not None:
            torch.backends.cudnn.deterministic=True
            torch.manual_seed(self.random_state)
        self.history
        self._commit = {}
        self.epochid = 0
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lowest_score=None
        self.highest_score=None

    def set_data(self, db=None, train_dl=None, valid_dl=None, device=None):
        if db:
            assert train_dl is None, 'You cannot specify both data and train_dl'
            assert valid_dl is None, 'You cannot specify both data and valid_dl'
            assert db.train_dl, 'The databunch you provided must have a train_dl'
            assert db.valid_dl, 'The databunch you provided must have a valid_dl'
            self.data = db
        elif not valid_dl:
            assert train_dl, 'No data provided'
            self.data = databunch(train_dl, train_dl, device=device)
        else:
            assert train_dl, 'No data provided'
            self.data = databunch(train_dl, valid_dl, device=device)
        try:
            self.device = device if device else self.data.device
        except:
            try:
                self.device = next(iter(self.train_dl)).device
            except:
                self.device = next(iter(self.train_dl))[0].device

    def __repr__(self):
        return 'Trainer( ' + self.model + ')'

    def to(self, device):
        self.device = device
        try:
            del self._optimizer
        except: pass

    def cpu(self):
        self.to(torch.device('cpu'))

    def gpu(self):
        self.to(torch.device('cuda:0'))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.train_dl = self.data.train_dl
        self.valid_dl = self.data.valid_dl

    @property
    def min_lr(self):
        try:
            return self.lr[0]
        except:
            try:
                return self.lr
            except:
                return 1e-3

    @property
    def max_lr(self):
        try:
            return self.lr[1]
        except: pass
        try:
            return self.lr[0]
        except: pass
        return self.lr

    def set_optimizer_param(self, key, value):
        if value is not None:
            self._optimizerparams[key] = value
        else:
            try:
                del self._optimizerparams[key]
            except: pass
        try:
            del self._optimizer
            del self._scheduler
        except: pass

    @property
    def weight_decay(self):
        return self.optimizer.param_groups[0]['weight_decay']

    @weight_decay.setter
    def weight_decay(self, value):
        self.set_optimizer_param('weight_decay', value)

    @property
    def momentum(self):
        return self.optimizer.param_groups[0]['betas']

    @momentum.setter
    def momentum(self, value):
        self.set_optimizer_param('betas', value)

    @property
    def optimizer(self):
        try:
            return self._optimizer
        except:
            self.set_optimizer_param('lr', self.min_lr)
            self._optimizer = self._optimizerclass(self.model.parameters(), **self._optimizerparams)
            return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizerclass = value
        try:
            del self._optimizer
            del self._scheduler
        except: pass

    def del_optimizer(self):
        try:
            del self._optimizer
            del self._schduler
        except: pass

    @property
    def scheduler(self):
        try:
            return self._scheduler
        except:
            if type(self.lr) is list:
                steps = int(round((len(self.train_dl) * self.cycle_epochs)))
                if self.schedulertype == 'cyclic':
                    self._scheduler = cyclicallr(self.optimizer, self.min_lr, self.max_lr, steps)
                elif self.schedulertype == 'onecycle':
                    self._scheduler = onecyclelr(self.optimizer, self.min_lr, self.max_lr, steps)
                else:
                    self._scheduler = uniformlr()
            else:
                self._scheduler = uniformlr()
            return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self.schedulertype = value
        try:
            del self._scheduler
        except: pass

    def change_lr(self, lr):
        try:
            if self.lr == lr:
                return
        except: pass
        self.lr = lr
        try:
            del self._scheduler
        except: pass

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
    def valid_ds(self):
        return self.train_dl.dataset

    @property
    def train_ds(self):
        return self.valid_dl.dataset

    @property
    def valid_tensors(self):
        return self.valid_dl.dataset.tensors

    @property
    def train_tensors(self):
        return self.train_dl.dataset.tensors

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

    def predict(self, *X):
        self.model.eval()
        X = [ x.to(self.model.device) for x in X ]
        return self.post_forward(self.model(*X))

    def post_forward(self, y):
        return y

    def list_commits(self):
        return self._commit.keys()

    def commit(self, label):
        "save the model and optimzer state, allowing to revert to a previous state"
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self._commit[label] = (model_state, optimizer_state, self._optimizerparams)

    def revert(self, label):
        "revert the model and optimizer to a previously commited state, deletes the commit point"
        if label in self._commit:
            model_state, optimizer_state, self._optimizerparams = self._commit.pop(label)
            self.model.load_state_dict(model_state)
            self.del_optimizer()            
            self.optimizer.load_state_dict(optimizer_state)
        else:
            print('commit point {label} not found')
    
    def checkout(self, label):
        "switches the model and optimizer to a previously commited state, keeps the commit point"
        if label in self._commit:
            model_state, optimizer_state, self._optimizerparams = self._commit[label]
            self.model.load_state_dict(model_state)
            self.del_optimizer()            
            self.optimizer.load_state_dict(optimizer_state)  
        else:
            print('commit point {label} not found')

    def remove_checkpoint(self, label):
        self._commit.pop(label)

    def purge(self, label):
        "switches the model and optimizer to a previously commited state, keeps only the commit point"
        if label in self._commit:
            self.checkout(label)
            self._commit = { l:s for l, s in self._commit.items() if l == label }
        else:
            print(f'commit point {label} not found')

    def train_correct(self):
        with ordered_dl(self.train_dl) as dl:
            for i, t in enumerate(dl):
                *X, y = [ a.to(self.model.device) for a in t ]
                y_pred = self.predict(*X)
                for ii, yy in zip(i, y == y_pred):
                    print(ii, yy)

    def validate_loss(self, dl=None):
        if not dl:
            dl = self.valid_dl
        with torch.set_grad_enabled(False):
            losses = []
            for t in dl:
                *X, y = [ a.to(self.model.device) for a in t ]
                losses.append((self.loss_xy(*X, y=y)[0].item(), len(y)))
            sums = [ sum(x) for x in zip(*losses) ]
            return sums[0] / sums[1]

    def validate(self, pbar=None):
        epoch = self.history.create_epoch('valid')
        epoch.report = True
        with torch.set_grad_enabled(False):
            epoch.before_epoch()
            for t in self.valid_dl:
                *X, y = [ a.to(self.model.device) for a in t ]
                epoch.before_batch(X, y)
                loss, y_pred = self.loss_xy(*X, y=y)
                epoch.after_batch( X, y, y_pred, loss )
                if pbar is not None:
                    pbar.update(self.valid_dl.batch_size)
            epoch.after_epoch()
        return epoch

    def loss_xy(self, *X, y=None):
        try:
            y_pred = self.model(*X)
        except:
            y_pred = self.model(X)
        return self.loss(y_pred, y), self.post_forward(y_pred)

    def train_batch(self, *X, y=None):
        self.optimizer.zero_grad()
        loss, y_pred = self.loss_xy(*X, y=y)
        loss.backward()
        self.optimizer.step()
        return loss, y_pred
    
    def train(self, epochs, lr=None, report_frequency=None, save=None, optimizer=None, weight_decay=None, momentum=None, save_lowest=None, save_highest=None):
        if save:
            self.save = save
        if weight_decay is not None and self.weight_decay != weight_decay:
            self.weight_decay = weight_decay
            self.del_optimizer()
        if momentum is not None and self.momentum != momentum:
            self.momentum = momentum
            self.del_optimizer()
        if optimizer and self._optimizerclass != optimizer:
            self.del_optimizer()
            self._optimizerclass=optimizer
        if report_frequency is None:
            report_frequency = self.report_frequency
        if lr:
            self.change_lr(lr)
        model = self.model
        model.train()
        reports = math.ceil(epochs / report_frequency)
        maxepoch = self.epochid + epochs
        batches = len(self.train_dl) * self.train_dl.batch_size * epochs + len(self.valid_dl) * self.valid_dl.batch_size * reports
        pbar = tqdm(range(batches), desc='Total', leave=False)
        for i in range(epochs):
            self.epochid += 1
            try:
                del self._scheduler
            except: pass
            self.scheduler
            epoch = self.history.create_epoch('train')
            if self.log and (((i + 1) % report_frequency) == 0 or i == epochs - 1):
                epoch.report = True
            epoch.before_epoch()
            for t in self.train_dl:
                *X, y = [ a.to(self.model.device) for a in t ]
                epoch.before_batch( X, y )
                loss, y_pred = self.train_batch(*X, y=y)
                self.scheduler.step()
                try:
                    y_pred = model.predict(y_pred)
                except: pass
                epoch.after_batch( X, y, y_pred, loss)
                pbar.update(self.train_dl.batch_size)
            epoch.after_epoch()
            if epoch.report:
                vepoch = self.validate(pbar = pbar)
                self.history.register_epoch(epoch)
                self.history.register_epoch(vepoch)
                print(f'{self.epochid} {epoch.time():.2f}s {epoch} {vepoch}')
                if save is not None:
                    self.commit(f'{save}-{self.epochid}')
                if save_lowest is not None:
                    if self.lowest_score is None or vepoch[save_lowest] < self.lowest_score:
                        self.lowest_score = vepoch[save_lowest]
                        self.commit('lowest')
                if save_highest is not None:
                    if self.highest_score is None or vepoch[save_highest] > self._highest_score:
                        self.highest_score = vepoch[save_highest]
                        self.commit('highest')
    
    def lowest(self):
        self.checkout('lowest')

    def highest(self):
        self.checkout('highest')

    def tune(self, params,setter, lr=[1e-6, 1e-2], steps=40, smooth=0.05, label=None, **kwargs):
        lr_values = exprange(*lr, steps)
        if label is None:
            label = str(setter)
        if len(params) == 2:
            params = range3(*params)
        with tuner(self, lr_values, self.set_lr, smooth=0.05, label=label) as t:
            t.run_multi(params, setter)

    def tune_weight_decay(self, lr=[1e-6,1e-4], params=[1e-6, 1], steps=40, smooth=0.05, yscale='log', **kwargs):
        self.tune( params, partial(self.set_optimizer_param, 'weight_decay'), lr=lr, steps=steps, smooth=smooth, label='weight decay', yscale=yscale, **kwargs)

    def lr_find(self, lr=[1e-6, 10], steps=100, smooth=0.05, **kwargs):
        with tuner(self, exprange(lr[0], lr[1], steps), self.set_lr, label='lr', yscale='log', smooth=smooth) as t:
            t.run()

    def plot(self, *metric, **kwargs):
        self.history.plot(*metric, **kwargs)
        

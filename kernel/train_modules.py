import torch
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, f1_score
from .helper import *

class training_module():
    def __init__(self, history):
        self.history = history
        self.trainer = history.trainer
        
    def require_module(self, module):
        self.history.require_module(module)
    
    def requirements(self):
        return False
        
class store_y(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        if epoch['n'] == 0:
            epoch['y'] = y.cpu().numpy()
            epoch['y_pred'] = y_pred.cpu().numpy()
        else:
            epoch['y'] = np.vstack([epoch['y'], y.cpu().numpy()])
            epoch['y_pred'] = np.vstack([epoch['y_pred'], y_pred.cpu().numpy()])     

class store_contingencies(training_module):
    "stores a 2x2 contingency table"
    def requirements(self):
        assert self.trainer.out_features == 1

    def after_batch(self, epoch, X, y, y_pred, loss):
        confusion_vector = torch.round(y_pred) / torch.round(y)
        epoch['tp'] += torch.sum(confusion_vector == 1).item()
        epoch['fp'] += torch.sum(confusion_vector == float('inf')).item()
        epoch['tn'] += torch.sum(torch.isnan(confusion_vector)).item()
        epoch['fn'] += torch.sum(confusion_vector == 0).item()

class store_confusion(training_module):
    "stores the entire confusion matrix, 1hot encoding must be decoded"
    def requirements(self):
        assert self.trainer.out_features > 1

    def after_batch(self, epoch, X, y, y_pred, loss):
        classes = self.trainer.out_features
        yt = y.data.cpu().numpy()
        if len(y_pred.shape) > 1: # check for 1-hot encoding
            y_pred = y_pred.max(1)[1]
        yc = y_pred.data.cpu().numpy()
        cm = confusion_matrix(yt, yc, range(classes))
        epoch['cm'] = cm + epoch['cm']

class store_sse(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        epoch['sse'] += torch.sum((y_pred - y)**2).item()

class store_sspe(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        yy = y > 0
        epoch['sspen'] += torch.sum(yy).item()
        epoch['sspe'] += torch.sum(torch.where(yy,((y_pred - y)/y)**2, yy.float())).item()

class store_f1(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        y = to_numpy(y)
        epoch['f1'] += f1_score(y, to_numpy(y_pred)) * len(y)

class store_sae(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        epoch['sae'] += torch.sum(torch.abs(y_pred - y)).item()

class store_r2(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        y = to_numpy(y)
        epoch['r2'] += r2_score(y, to_numpy(y_pred)) * len(y)

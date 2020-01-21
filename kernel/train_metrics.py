from .train_modules import *
from sklearn.metrics import accuracy_score
import math

def name_metrics(metrics):
    for i, m in enumerate(metrics):
        try:
            m, m.__name__ = m.value, m.__name__
        except:
            m.__name__ = f'm_{i}'
        yield m

class train_metrics():
    def __init__(self, history=None):
        self.history = history
        
    def requirements(self):
        pass
    
    @property
    def __name__(self):
        return self.__name()

    def __name(self):
        return self.__class__.__name__
  
    @staticmethod
    def value(epoch):
        pass
        
class acc(train_metrics):
    def requirements(self):
        self.history.require_module(store_contingencies)
    
    @staticmethod
    def value(epoch):
        return (epoch['tp'] + epoch['tn']) / epoch['n']

class acc_mc(train_metrics):
    def requirements(self):
        self.history.require_module(store_confusion)
    
    @staticmethod
    def value(epoch):
        return np.diag(epoch['cm']).sum() / epoch['n']

class recall(train_metrics):
    def requirements(self):
        self.history.require_module(store_contingencies)

    @staticmethod
    def value(epoch):
        try:
            return epoch['tp'] / (epoch['tp'] + epoch['fn']) 
        except:
            return 0

class precision(train_metrics):
    def requirements(self):
        self.history.require_module(store_contingencies)

    @staticmethod
    def value(epoch):
        try:
            return epoch['tp'] / (epoch['tp'] + epoch['fp']) 
        except:
            return 0

class f1(train_metrics):
    def requirements(self):
        self.history.require_module(store_f1)

    @staticmethod
    def value(epoch):
        return epoch['f1'] / epoch['n']

class mse(train_metrics):
    def requirements(self):
        self.history.require_module(store_sse)

    @staticmethod
    def value(epoch):
        return epoch['sse'] / epoch['n']

class rmse(train_metrics):
    def requirements(self):
        self.history.require_module(store_sse)

    @staticmethod
    def value(epoch):
        return math.sqrt(epoch['sse'] / epoch['n'])

class rmspe(train_metrics):
    def requirements(self):
        self.history.require_module(store_sspe)

    @staticmethod
    def value(epoch):
        return math.sqrt(epoch['sspe'] / epoch['sspen'])

class r2(train_metrics):
    def requirements(self):
        self.history.require_module(store_r2)

    @staticmethod
    def value(epoch):
        return epoch['r2'] / epoch['n']


class loss(train_metrics):
    @staticmethod
    def value(epoch):
        return epoch['loss']    
    

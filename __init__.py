from .kernel.gpu import *
from .kernel.trainer import *
from .kernel.train_modules import *
from .kernel.train_diagnostics import *
from .kernel.jcollections import *
from .kernel.transfer import *
from .kernel.perceptron import *
from .version import __version__
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import numpy as np
import pandas as pd
import pipetorch as pt

def list_all(s):
    try:
        return s.__all__
    except:
        return [ o for o in dir(s) if not o.startswith('_') ]

#subpackages = [ jtorch.train, jtorch.train_modules, jtorch.train_diagnostics, jtorch.train_metrics, jtorch.jcollections ]

subpackages = [ trainer ]

#__all__ = [ f for s in subpackages for f in list_all(s) ]


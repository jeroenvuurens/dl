import pandas as pd
import numpy as np
from fastai.vision import *

""" A Databunch is used in FastAI to wrap datasets for training and validation. A Databunch should provide dataloaders for both via train_dl and valid_dl.

Here we use the FastAI vision databunch, which takes a path in which the images are stored in folders that indicate their class.
""" 

def sample(self, device=None):
    X, y = self.one_batch()
    if device is not None:
        return X.to(device), y.to(device)
    return X, y

def image_databunch(path, size, valid_perc=0.2, tfms=None, num_workers=0, batch_size=16):
    if tfms == None:
        tfms = get_transforms()
    data = ImageList.from_folder(path)\
            .split_by_rand_pct(valid_pct=valid_perc)\
            .label_from_folder()\
            .transform(tfms, size=size)\
            .databunch(bs = batch_size, num_workers = num_workers)
    data.sample = partial(sample, data)
    data.inverse_transform_y = lambda x:x
    return data
    #return ImageDataBunch.from_folder(path, ds_tfms=tfms, valid='valid', test='test', size=size, bs = config['batch_size'], device = config['device'])

def text_image_databunch(path, size, valid_perc=0.2, num_workers=0, batch_size=64):
    """ Specific for text images, we have to get rid of the flip transformation, because it does not apply to text 
    """
    tfms = get_transforms(do_flip=False)

    data = image_databunch(path, size, valid_perc=valid_perc, tfms=tfms, num_workers=num_workers, batch_size=batch_size)
    data.train_X = data.train_ds.x
    data.valid_X = data.valid_ds.x
    data.train_y = data.train_ds.y
    data.valid_y = data.valid_ds.y
    data.sample = partial(sample, data)
    data.inverse_transform_y = lambda x:x
    return data

def cifar(size=32, batch_size=32):
    path='/data/datasets/cifar10'
    data = ImageDataBunch.from_folder(path, valid='test', bs=batch_size, size=size).normalize(cifar_stats)
    data.sample = data.one_batch 
    data.inverse_transform_y = lambda x:x
    return data

from fastai.text import *
import html
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.datasets import IMDB, Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.vocab import Vectors
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import spacy
import dill as pickle
import random
import math
import time
import re

tokenizer = get_tokenizer("spacy")
identity=lambda x:x

def save_to_pickle(dataSetObject,PATH):
    with open(PATH,'wb') as output:
        for i in dataSetObject:
            pickle.dump(vars(i), output, pickle.HIGHEST_PROTOCOL)

def load_pickle(PATH, FIELDNAMES, FIELD):
    dataList = []
    with open(PATH, "rb") as input_file:
        while True:
            try:
                # Taking the dictionary instance as the input Instance
                inputInstance = pickle.load(input_file)
                # plugging it into the list
                dataInstance =  [inputInstance[FIELDNAMES[0]],inputInstance[FIELDNAMES[1]]]
                # Finally creating an example objects list
                dataList.append(Example().fromlist(dataInstance,fields=FIELD))
            except EOFError:
                break

    # At last creating a data Set Object
    exampleListObject = Dataset(dataList, fields=data_fields)
    return exampleListObject 

class TextData:
    def __init__(self, train, valid=None, batch_size=32,
                 sort_within_batch=False, repeat=False, sort=False, shuffle=False, y_field_ind=1, cache_valid=False):
        self.train_ds = train
        self.valid_ds = valid
        self.vectors = vectors
        self.sortkey = sortkey
        self.cache_valid = cache_valid
        self.batch_size = batch_size 
        self.sort_within_batch = sort_within_batch
        self.y_field_ind = y_field_ind
        self.set_fields(fields)
        self.sort = sort
        self.repeat = repeat
        self.shuffle = shuffle

    def set_fields(self, fields):
        self.fields = fields
        self.x_field = fields[1-self.y_field_ind][0]
        self.y_field = fields[self.y_field_ind][0]
        for n, f in self.fields:
            self.__setattr__(n, f)

    def save(self, path):
        for n, f in self.fields:
            pickle.dump(f, open(f'{path}/{n}.field.pkl', 'wb'))
        save_to_pickle(self.train_ds, f'{path}/train.ds')
        save_to_pickle(self.valid_ds, f'{path}/valid.ds')

    def build_vocab(self):
        for n, f in self.fields:
            if f.use_vocab:
                f.build_vocab(self.train_ds, self.valid_ds)
                if self.vectors is not None:
                    try:
                        path, name = self.vectors.rsplit('/', 1)
                        vectors = Vectors(name, cache=path)
                    except:
                        vectors = self.vectors
                    f.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

    def _createdl(self, ds, **kwargs):
        train = ds == self.train_ds
        if train:
            args = {'sort_key':self.sortkey, 'batch_size':self.batch_size, 'sort_within_batch':self.sort_within_batch, 'repeat':self.repeat, 'sort':self.sort, 'shuffle':self.shuffle, 'train':True}
            args.update(kwargs)
        else:
            args = {'batch_size':self.batch_size, 'repeat':self.repeat, 'shuffle':False, 'train':False, 'sort':False}
            args.update(kwargs)
        args = { k:v for k, v in args.items() if v is not None}
        dl = BatchGenerator(BucketIterator(ds, **args), self.x_field, self.y_field)
        dl.dataset = ds
        return dl

    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            self._train_dl = self._createdl(self.train_ds)
            return self._train_dl

    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            if self.cache_valid:
                dl = self._createdl(self.valid_ds, batch_size=len(self.valid_ds))
                self._valid_dl = [ next(iter(dl)) ]
            else:
                self._valid_dl = self._createdl(self.valid_ds)
            return self._valid_dl

    @classmethod
    def from_csv(cls, fields, filename, valid_perc=0.8, random_state=None, skip_header=True, **kwargs):
        train, valid = TabularDataset( path=filename, format='csv', skip_header=skip_header, fields=fields).split(1-valid_perc)
        return cls(fields, train, valid, **kwargs)

    @classmethod
    def load(cls, fields, filename, path, valid_perc=0.8, random_state=None, skip_header=True, **kwargs):
        fields = [ (n, pickle.load(open(f'{path}/{n}.field.pkl', 'rb'))) for n, _ in fields ]
        train, valid = TabularDataset( path=filename, format='csv', skip_header=skip_header, fields=fields).split(1-valid_perc)
        return cls( fields, train, valid, **kwargs ) 

    @classmethod
    def from_datasets(cls, train_ds, valid_ds):
        data = cls(train_ds, valid_ds)
        return data

def imdb(tokenizer=None):
    if not tokenizer:
        tokenizer=get_tokenizer("spacy")
    train_ds, valid_ds = IMDB(tokenizer=tokenizer) 
    return TextData.from_datasets(train_ds, valid_ds)



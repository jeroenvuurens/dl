from fastai.text import *
import html
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.vocab import Vectors
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe

import spacy
import dill as pickle
import random
import math
import time
import re

identity=lambda x:x

def stripper(regex):
    regex = re.compile(regex)
    def strip(text):
        return re.sub(regex, ' ', text) # remove section
    return strip

tag_stripper = stripper('<.*?>')

def boolean_label(positive):
    def tokenize(text):
        return 1 if text == positive else 0
        #return (0.0, 1.0) if text == positive else (1.0, 0.0)
    return tokenize

def tokenizer(lang='en', reverse=False, pre=identity, post=identity):
    tok = spacy.load(lang, disable=["tagger", "parser", "ner", "textcat"])
    if reverse:
        def tokenize(text):
            return [post(t.text) for t in tok.tokenizer(pre(text))][::-1]
    else:
        def tokenize(text):
            return [post(t.text) for t in tok.tokenizer(pre(text))]
    return tokenize

class TextField(Field):
   def __init__(self, lang, sos='<sos>', eos='<eos>', lower=True, vectors=None, include_lengths=False, **kwargs):
       self.tokenizer = tokenizer(lang, **kwargs)
       super().__init__(tokenize = self.tokenizer, init_token=sos, eos_token=eos, lower=lower, include_lengths=include_lengths)

class LabelField(Field):
    def __init__(self, dtype=torch.long, **kwargs):
        super().__init__(sequential=False, use_vocab=False, pad_token=None,  unk_token=None, dtype=dtype, **kwargs)

class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)

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
    def __init__(self, fields, train, valid=None, vectors=None, sortkey=None, batch_size=32,
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
        self.inverse_transform_y = lambda y : y

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



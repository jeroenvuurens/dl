import pandas as pd
import numpy as np
import seaborn as sns
import torch
import pickle
import matplotlib.pyplot as plt
from ..kernel.helper import *
from ..kernel.data import *
from pandas import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
from torch.utils.data import TensorDataset, Dataset, DataLoader, SubsetRandomSampler, Sampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
identity=lambda x:x

class SequenceScaler:
    def __init__(self, df):
        self.columns = df.columns
        self.scalers = { c:StandardScaler() for c in self.columns }
    
    def fit(self, df):
        for c in df.columns:
            if c in self.scalers:
                self.scalers[c].fit(df[[c]].dropna())
        return self
    
    def transform(self, df):
        for c in df.columns:
            if c in self.scalers:
                df.loc[:,c] = self.scalers[c].transform(df[[c]])
        return df
    
    def inverse_transform_y(self, y):
        return self.scalers[self.columns[-1]].inverse_transform(y)

class SequenceDatasetX(Dataset):
    def __init__(self, tensor, indices, window, shift):
        super().__init__()
        self.tensor = tensor
        self.window = window
        self.indices = indices

    def __getitem__(self, index):
        try:
            return self.tensor[self.indices[index]:self.indices[index]+self.window]
        except:
            return torch.cat([self.tensor[i:i+self.window].unsqueeze(0) for i in self.indices[index]])

    def __len__(self):
        return len(self.indices)

class SequenceDatasetY(Dataset):
    def __init__(self, tensor, indices, window, shift):
        super().__init__()
        self.tensor = tensor
        self.window = window
        self.shift = shift
        self.indices = indices

    def __getitem__(self, index):
        try:
            return self.tensor[self.indices[index]+self.window+self.shift-1, -1]
        except:
            return torch.cat([self.tensor[i+self.window+self.shift-1, -1:] for i in self.indices[index]]).view(-1)

    def __len__(self):
        return len(self.indices)

class SequenceDataFrame(DataFrame):
    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    @classmethod
    def from_csv(cls, *args, sep=',', **kwargs):
        df = pd.read_csv(*args, sep=sep, **kwargs)
        return cls(df)

    @property
    def _constructor(self):
        return SequenceDataFrame

    def to_dataset(self, window, split=0.2, scale=True, shift_y=1, columns=None, dtype=None):
        """
        Converts a SequenceDataFrame into SequenceDataSets
        split: the fractions for the validations an optionally test set (use a tuple with 2 values). The remainder is used for training.
        scale: True, scale all columns, but you can also list the columns to be scaled.
        window: The length of the time sequence in X, e.g. 7 means a sequence of 7 time steps is stored in X
        shift_y: How may timesteps ahead to predict y, default=1, e.g. predict the next value.
        """
        df = self[columns] if columns else self
        if dtype:
            df = df.astype(dtype)
        indices = np.where(df.iloc[window+shift_y-1:,-1].notnull())[0]
        indices = [ i for i in indices if not df.iloc[i:i+window, :-1].isnull().values.any() ]
        length = len(indices)
        try:
            train_indices = indices[:int((1 - sum(split))* length)]
            valid_indices = indices[int((1 - sum(split))* length):int((1 - split[1])* length)]
            test_indices = indices[int((1 - split[1])* length):]
        except:
            train_indices = indices[:int((1 - split)* length)]
            valid_indices = indices[int((1 - split) * length):]
            test_indices = []
        if len(train_indices) > 0 and (type(scale) == list or scale):
            cslice = slice(0, train_indices[-1] + window + shift_y - 2)
            if type(scale) == list:
                scale = [ s for s in scale if s in df.columns ]
                scale = df[scale]
            else:
                scale = df
            scaler = SequenceScaler(scale[cslice])
            scaler.fit(scale[cslice])
            array = scaler.transform(df)
        else:
            scaler = None
            array = df
        tensor_X = torch.from_numpy(array.to_numpy())
        tensor_y = tensor_X[:, -1:]
        tensor_X = tensor_X[:, :-1]

        p = [ SequenceDataset(tensor_X, tensor_y, train_indices, scaler, window, shift_y) ]
        p.append( SequenceDataset(tensor_X, tensor_y, valid_indices, scaler, window, shift_y) )
        if len(test_indices) > 0:
            p.append( SequenceDataset(tensor_X, tensor_y, test_indices, scaler, window, shift_y) )
        return p

    def groupby(self, by, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True):
        r = super().groupby(by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
        return GroupedSequenceDataFrame(r)

class GroupedSequenceDataFrame(DataFrameGroupBy):
    def __init__(self, data=None):
        super().__init__(obj=data.obj, keys=data.keys, axis=data.axis, level=data.level, grouper=data.grouper, exclusions=data.exclusions, 
                selection=data._selection, as_index=data.as_index, sort=data.sort, group_keys=data.group_keys,
                observed=data.observed, mutated=data.mutated, dropna=data.dropna)

    @property
    def _constructor(self):
        return GroupedSequenceDataFrame

    def to_dataset(self, window, split=0.2, scale=True, shift_y=1, columns=None, dtype=None):
        dss = []
        for key, group in self:
            dss.append(SequenceDataFrame(group).to_dataset(window, split=split, scale=scale, shift_y=shift_y, columns=columns, dtype=dtype))
        
        return [ConcatDataset(ds) for ds in zip(*dss)]

class SequenceDataset(Dataset):
    def __init__(self, tensor_X, tensor_y, indices, scaler, window, shift):
        super().__init__()
        self.indices = indices
        self.X = SequenceDatasetX(tensor_X, self.indices, window, shift )
        self.y = SequenceDatasetY(tensor_y, self.indices, window, shift )
        self.scaler = scaler
        self.window = window
        self.shift = shift
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.indices)

    def inverse_transform_y(self, y):
        if self.scaler:
            return self.scaler.inverse_transform_y(y)
        return y

def flights_df():
    df = sns.load_dataset('flights')
    df['month'] = df.month.map({'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'Jun':5, 'Jul':6, 'Aug':7, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}).astype(np.float32)
    df['passengers'] = df.passengers.astype(np.float32)
    df['year'] = df.year.astype(np.float32)
    return SequenceDataFrame(df)
    
    

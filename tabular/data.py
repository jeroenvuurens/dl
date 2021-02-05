import pandas as pd
import numpy as np
import seaborn as sns
import torch
import pickle
import matplotlib.pyplot as plt
from ..kernel.helper import *
from torch.utils.data import TensorDataset, Dataset, DataLoader, SubsetRandomSampler, Sampler
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.compose import ColumnTransformer
identity=lambda x:x

def map_categories(df, column):
    df = df[column]
    try:
        s = sorted(df.unique())
    except:
        s = df.unique()
    d = { v:i for i, v in enumerate(s) }
    df = df.map(d).astype(np.long)
    return df

class Scaler:
    def __init__(self, *arrays, scale=True, scale_y=True):
        if scale:
            scalers = [ Scaler._scaler(a) for a in arrays[:-1] ]
        else:
            scalers = [None] * (len(arrays) - 1)
        if scale_y:
            scalers.append( Scaler._scaler(arrays[-1]) )
        else:
            scalers.append(None)
        self.scalers = tuple( scalers )

    @staticmethod
    def _scaler(arr):
        if arr.dtype.kind == 'f':
            min = arr.min()
            max = arr.max()
            if max - min < 0.1 or min < -2 or max > 2:
                scaler = StandardScaler()
                if len(arr.shape) == 1:
                    arr = arr.reshape(-1, 1)
                scaler.fit(arr)
                return scaler

    def transform(self, *arrays):
        out = []
        for a, scaler in zip(arrays, self.scalers):
            if scaler is not None:
                if len(a.shape) == 1:
                    a = a.reshape(-1, 1)
                    a = scaler.transform(a)
                    a = a.reshape(-1)
                else:
                    a = scaler.transform(a)
                out.append(a)
            else:
                out.append(a)
        return out

    def inverse_transform_y(self, y):
        if self.scalers[-1] is not None:
            y = to_numpy(y)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
                y = torch.tensor(self.scalers[-1].inverse_transform(y))
                y = y.reshape(-1)
            else:
                y = torch.tensor(self.scalers[-1].inverse_transform(y))
        return y

    def inverse_transform_x(self, *x):
        transform = [ x[i] if self.scalers[i] is None else torch.tensor(self.scalers[i].inverse_transform(to_numpy(x[i]))) for i in range(len(x)) ]
        if len(transform) == 1:
            return transform[0]
        return transform

    def inverse_transform(self, sample):
        s = self.inverse_transform_x(sample[:-1])
        s.append( self.inverse_transform_y( sample[-1] ) )
        return s

class ArrayDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *arrays, scale_y = True, scale = True, scaler = None, transform = True):
        assert all(arrays[0].shape[0] == a.shape[0] for a in arrays)
        if scaler:
            self.scaler = scaler
        else:
            self.scaler = Scaler( *arrays, scale=scale, scale_y = scale_y )
        if transform:
            arrays = self.scaler.transform( *arrays )
        self.tensors = [ torch.from_numpy(a) for a in arrays ]
        super().__init__(*self.tensors)

    def to(self, device):
        self.tensors = [ t.to(device) for t in self.tensors ]

    def inverse_transform_y(self, y):
        return self.scaler.inverse_transform_y(y)

    def inverse_transform_x(self, *x):
        return self.scaler.inverse_transform_x(*x)

    def inverse_transform(self, *sample):
        return self.scaler.inverse_transform( *sample )

    def X(self, indices):
        if len(self.tensors) == 2:
            return self.tensors[0][indices]
        else:
            return [ t[indices] for t in self.tensors[:-1] ]

    def y(self, indices):
        return self.tensors[-1][indices]

class CompleteSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class TabularDatabunch:
    def __init__(self, ds, train_indices, valid_indices, batch_size=32, valid_batch_size=None, num_workers=0, shuffle=True, device=torch.device('cuda:0'), gpu=None, pin_memory=False):
        self.ds = ds
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size if valid_batch_size else len(self.valid_indices)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.to(device, gpu)

    def to(self, device, gpu=None):
        if gpu is not None:
            if gpu < 0:
                device = torch.device('cpu')
            else:
                device = torch.device('cuda:' + str(gpu))
        self.ds.to(device)
        self.device = device
        return self

    def cpu(self):
        return self.to(torch.device('cpu'))

    def gpu(self):
        return self.to(torch.device('cuda:0'))

    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            self._train_dl = self._dl(self.train_indices, self.batch_size, self.num_workers, self.shuffle)
            return self._train_dl

    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            self._valid_dl = self._dl(self.valid_indices, self.valid_batch_size, self.num_workers, shuffle=False)
            return self._valid_dl

    def _dl(self, indices, batch_size, num_workers=0, shuffle=False):
        if shuffle:
            sampler = SubsetRandomSampler(indices)
            return DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=self.pin_memory)
        else:
            sampler = CompleteSampler(indices)
            return DataLoader(self.ds, batch_size=batch_size, num_workers=0, sampler=sampler, pin_memory=self.pin_memory)


    @property
    def train_numpy(self):
        return tuple( to_numpy(a[self.train_indices]) for a in self.ds.tensors )

    @property
    def valid_numpy(self):
        return tuple( to_numpy(a[self.valid_indices]) for a in self.ds.tensors )

    def inverse_transform_y(self, y):
        return self.ds.inverse_transform_y(y)

    def inverse_transform(self, *arrays):
        return self.ds.inverse_transform(*arrays)

    def inverse_transform_x(self, *x):
        #if len(x) == 1:
        #    return self.ds.inverse_transform_x(x[0])
        return self.ds.inverse_transform_x(*x)

    @staticmethod
    def _split(arrays, valid_perc, shuffle=True):
        dataset_size = len(arrays[0])
        indices = np.array(range(dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        if valid_perc == 0.0:
            return indices, indices
        split = int(np.floor(valid_perc * dataset_size))
        return indices[split:], indices[:split]

    @staticmethod
    def _balance_y(y, train_indices):
        y = y[train_indices]
        indices = [np.where(y==l)[0] for l in np.unique(y)]
        classlengths = [len(i) for i in indices]
        n = max(classlengths)
        mask = np.hstack([np.random.choice(i, n-l, replace=True) for l,i in zip(classlengths, indices)])
        indices = np.hstack([mask, range(len(y))])
        return train_indices[indices]

    @staticmethod
    def _polynomials(arrays, degree):
        arr = []
        for a in arrays[:-1]:
            if a.dtype.kind == 'f':
                poly = PolynomialFeatures(degree, include_bias=False)
                arr.append(poly.fit_transform(a))
            else:
                arr.append(a)
        arr.append(arrays[-1])
        return arr

    @classmethod
    def from_arrays(cls, *arrays, train_indices = None, batch_size=32, scale=True, balance=False, degree=1, num_workers=0, random_seed=0, valid_perc=0.2, scale_y=True, shuffle=True, **kwargs):
        np.random.seed(random_seed)
        if degree > 1:
            arrays = cls._polynomials(arrays, degree)
        if train_indices:
            valid_indices = [ i for i in range(len(arrays[0])) if i not in train_indices ]
        else:
            train_indices, valid_indices = cls._split(arrays, valid_perc, shuffle)
        if balance:
            train_indices = cls._balance_y(arrays[-1], train_indices)
        return cls.from_split_arrays(train_indices, valid_indices, *arrays, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, scale=scale, scale_y=scale_y, **kwargs)

    @classmethod
    def from_split_arrays(cls, train_indices, valid_indices, *arrays, batch_size=32, scale=True, num_workers=0, scale_y=True, **kwargs):
        tarrays = [ a[train_indices] for a in arrays ]
        scaler = Scaler( *tarrays, scale=scale, scale_y=scale_y)
        ds = ArrayDataset( *arrays, scaler = scaler )
        return TabularDatabunch(ds, train_indices, valid_indices, batch_size=batch_size, num_workers=num_workers, **kwargs)

    @classmethod
    def from_list( cls, l, dtype=np.float32, valid_perc=0.0, scale=False, **kwargs ):
        """ Mostly used for demonstrating binary operators, therefore, scale 
        and balance are turned off by default
        """
        t = np.array(l, dtype=dtype)
        X = t[:, :-1]
        y = t[:, -1]
        return cls.from_arrays(X, y, valid_perc=valid_perc, scale=scale, **kwargs)

    @classmethod
    def from_df( cls, df, target, *features, batch_size=64, balance=False, **kwargs ):
        features = expand_features(df, target, *features)
        arrays = cls._df_split_xy(df, target, *features)
        return cls.from_arrays(*arrays, batch_size=batch_size, balance=balance, **kwargs)

    @staticmethod
    def _df_split_xy(df, target, *features):
        a = []
        featuresf = [c for c in features if df[c].dtype.kind == 'f']
        if len(featuresf) > 0:
            a.append(df[featuresf].to_numpy())
        featuresl = [c for c in features if df[c].dtype.kind in {'i', 'u'}]
        if len(featuresl) > 0:
            a.append(df[featuresl].to_numpy())
        a.append( df[target].to_numpy() )
        #print([ (df[c].dtype, df[c].dtype.kind) for c in features ])
        #if len(a[0].shape) < 2:
        #    a[0] = a[0].reshape(a[0].shape[0], 1)
        return a

    def reset(self):
        try:
            del self._valid_dl
        except: pass
        try:
            del self._train_dl
        except: pass

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = min(value, len(self.train_indices))
        self.reset()

    @property
    def train_X(self):
        return self.ds.X(self.train_indices)

    @property
    def train_y(self):
        return self.ds.y(self.train_indices)

    @property
    def valid_X(self):
        return self.ds.X(self.valid_indices)

    @property
    def valid_y(self):
        return self.ds.y(self.valid_indices)

    @property
    def train_X_orig(self):
        return self.inverse_transform_x(self.train_X)

    @property
    def valid_X_orig(self):
        return self.inverse_transform_x(self.valid_X)

    @property
    def train_y_orig(self):
        return self.inverse_transform_y(self.train_y)

    @property
    def valid_y_orig(self):
        return self.inverse_transform_y(self.valid_y)

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        self._num_workers = value
        self.reset()

    def sample(self, device=None):
        arrays = next(iter(self.train_dl))
        if device is not None:
            arrays = [ a.to(device) for a in arrays ]
        return arrays

    def _plot(self, **kwargs):
        figure = plt.figure(**kwargs)
        return plt.axes()

    def plot_train(self, ind_x=None, **kwargs):
        p = self._plot(**kwargs)
        if ind_x is None:
            p.scatter(self.train_X_orig, self.train_y_orig)
        else:
            p.scatter(self.train_X_orig[:,ind_x], self.train_y_orig)
        return p

    def plot_valid(self, ind_x=None, **kwargs):
        p = self._plot(**kwargs)
        if ind_x is None:
            p.scatter(self.valid_X_orig, self.valid_y_orig)
        else:
            p.scatter(self.valid_X_orig[:,ind_x], self.valid_y_orig)
        return p

    def scatter_train(self, x1=0, x2=1, s=1, figsize=(8,8), **kwargs):
        p = self._plot(figsize=figsize, **kwargs)
        x_0 = self.train_X_orig[self.train_y == 0]
        x_1 = self.train_X_orig[self.train_y == 1]
        p.scatter(x_0[:, x1], x_0[:, x2], marker='x', label=0, s=s)
        p.scatter(x_1[:, x1], x_1[:, x2], marker='+', label=1, s=s)
        p.legend()
        return p

def expand_features(df, target, *features):
    if len(features) == 0:
        return [c for c in df.columns if c != target]
    else:
        return [c for c in features if c != target]

def read_csv(filename, nrows=100, drop=None, columns=None, dtype=dict(), intcols=[], **kwargs):
    df = pd.read_csv(filename, nrows=nrows, **kwargs)
    if drop:
        df = df.drop(columns=drop)
    elif columns:
        df = df[columns]
    float_cols = [c for c in df if df[c].dtype.kind == "f" or df[c].dtype.kind == "i"]
    float32_cols = {c:np.float32 for c in float_cols}
    float32_cols.update({ c:np.int64 for c in intcols })
    float32_cols.update(dtype)
    df = pd.read_csv(filename, dtype=float32_cols, **kwargs)
    if drop:
        df = df.drop(columns=drop)
    elif columns:
        df = df[columns]
    return df

def wines_pd():
    return read_csv('/data/datasets/winequality-red.csv', delimiter=';')

def wines(target, *features, valid_perc=0.2, threshold=None, **kwargs):
    df = wines_pd()
    if threshold is not None:
        df['quality'] = ((df.quality >= threshold) * 1.0).astype(np.float32)
    return TabularDatabunch.from_df( df, target, *features, valid_perc=valid_perc,  **kwargs )

def telco_churn_df():
    columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
    return read_csv('/data/datasets/telco_churn.csv', columns=columns)

def churn_df(one_hot=True):
    df = pd.read_csv('/data/datasets/Churn_Modelling.csv')
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    #df = df[['Age', 'Balance', 'IsActiveMember', 'Exited']]

    dfs = [df]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            if len(df[c].unique()) == 2 or not one_hot:
                df[c] = df[c].astype('category').cat.codes
            else:
                d = pd.get_dummies(df[c], prefix=c)
                dfs.append( d[d.columns[:-1]] )
                df.drop(columns=[c], inplace=True)
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)

    df = pd.concat(dfs, axis=1)
    df.Exited = df.Exited.astype(np.float32)
    return df

def churn(*features, balance=False, dtype=None, categories=None, one_hot=True, **kwargs):
    df = churn_df(one_hot=one_hot)
    if categories:
        for c in df.columns:
            if c not in categories:
                df[c] = df[c].astype(np.float32)
            else:
                df[c] = df[c].astype(np.long)
    elif dtype:
        df = df.astype(dtype)
    return TabularDatabunch.from_df(df, 'Exited', *features, balance=balance, **kwargs)

def movielens_df():
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    return read_csv("/data/datasets/ml-1m/ratings.dat",sep='::', intcols=['user_id', 'movie_id'], names=COLS).drop(columns=['timestamp'])

def movielens_movies_df():
    COLS = ['movie_id', 'title', 'genre']
    return pd.read_csv("/data/datasets/ml-1m/movies.dat",sep='::', names=COLS, engine='python')

def movielens(valid_perc=0.2, dtype=np.float32, **kwargs):
    df = movielens_df()
    return TabularDatabunch.from_df( df, 'rating', 'user_id', 'movie_id', valid_perc=valid_perc, scale=False, scale_y=False, **kwargs)

def dam(**kwargs):
    with open("/data/datasets/dam_water_data.pickle", "rb") as myfile:
        X_train, X_val, X_test, X_all, y_train, y_val, y_test, y_all = pickle.load(myfile)
        train_indices = [ i for i, v in enumerate(X_all) if (X_train == v).any() ]
        X_all = X_all.astype(np.float32)
        y_all = y_all.astype(np.float32)
    return TabularDatabunch.from_arrays(X_all, y_all, train_indices=train_indices, **kwargs)

def boston_housing_prices_df():
    """
    Load the Boston Housing Prices dataset and return it as a Pandas Dataframe
    """
    boston = load_boston()
    df = pd.DataFrame(boston['data'] )
    df.columns = boston['feature_names']
    df['PRICE']= boston['target']
    return df.astype(np.float32)

def boston_housing_prices_descr():
    print(load_boston().DESCR)

def boston(target='PRICE', *features, valid_perc=0.2, **kwargs):
    df = boston_housing_prices_df()
    return TabularDatabunch.from_df( df, target, *features, valid_perc=valid_perc, **kwargs)

def banking_df():
    df = read_csv("/data/datasets/bank-additional-full.csv", sep=';')
    return df

def banking_cleaned():
    df = banking_df()
    df['y'] = np.where(df.y == 'yes', 1, 0)
    columns = ['marital', 'job', 'poutcome', 'contact', 'housing', 'loan', 'default', 'education', 'day_of_week', 'month']
    for c in columns:
        df[f'{c}_ordinal'] = df[c].astype('category').cat.codes
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

def banking_desc():
    with open('/data/datasets/bank-additional-names.txt', 'r') as fin:
        print(''.join(fin.readlines()))

def banking(*features, valid_perc=0.2, **kwargs):
    df = banking_cleaned()
    return TabularDatabunch.from_df( df, 'y', *features, valid_perc=valid_perc, **kwargs)

def rossmann_orig_df():
    dtype = { 'Store': 'int', 'DayOfWeek': 'int', 'Sales': 'int', 'Customers': 'int', 'Open': 'int', 'Promo': 'int', 'SchoolHoliday': 'int', 'DayOfWeek': 'int', 'StateHoliday':'str', 'Date':'str'}
    df = read_csv("/data/datasets/rossmann/train.csv", delimiter=',', low_memory=True, engine='c', dtype=dtype, parse_dates=['Date'])
    return df

def rossmann_df(dtype=None, categories=None):
    df = rossmann_orig_df()
    df = df.drop(columns='Customers')
    df['dayofweek_cat'] = map_categories(df, 'DayOfWeek')
    df['stateholiday_cat'] = map_categories(df, 'StateHoliday')
    df['Date'] = pd.to_datetime(df.Date)
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    df.drop(columns=['StateHoliday', 'Date'], inplace=True)
    df.Sales = df.Sales.astype(np.float32)
    if categories:
        for c in df.columns:
            if c in categories:
                df[c] = df[c].astype(np.long)
            else:
                df[c] = df[c].astype(np.float32)
    elif dtype:
        df = df.astype(dtype)
    return df

def rossmann(*features, valid_perc=0.2, scale=True, dtype=None, categories=None, **kwargs):
    df = rossmann_df(dtype=dtype, categories=categories)
    return TabularDatabunch.from_df( df, 'Sales', *features, valid_perc=valid_perc, scale=scale, **kwargs)

def california_df():
    df = read_csv("/data/datasets/california.csv", sep=',')
    return df

def california_cleaned_df(*features, valid_perc=0.2, **kwargs):
    df = california_df()
    df['near_bay'] = np.where(df.ocean_proximity == 'NEAR BAY', 1, 0)
    df['near_ocean'] = np.where(df.ocean_proximity == 'NEAR OCEAN', 1, 0)
    df['island'] = np.where(df.ocean_proximity == 'ISLAND', 1, 0)
    df['inland'] = np.where(df.ocean_proximity == 'INLAND', 1, 0)
    df.drop(columns=['ocean_proximity'], inplace=True)
    return df

def california_lookup_df(*features, valid_perc=0.2, **kwargs):
    df = california_df()
    df['longitude_cat'] = map_categories(df, 'longitude')
    df['latitude_cat'] = map_categories(df, 'latitude')
    df['ocean_proximity_cat'] = map_categories(df, 'ocean_proximity')
    df['housing_median_age_cat'] = map_categories(df, 'housing_median_age')
    df.drop(columns=['ocean_proximity', 'housing_median_age'], inplace=True)
    return df

def california(*features, valid_perc=0.2, **kwargs):
    df = california_cleaned_df()
    return TabularDatabunch.from_df( df, 'median_house_value', *features, valid_perc=valid_perc, **kwargs)

def california_lookup(*features, valid_perc=0.2, **kwargs):
    df = california_lookup_df()
    return TabularDatabunch.from_df( df, 'median_house_value', *features, valid_perc=valid_perc, **kwargs)

def flights_df():
    df = sns.load_dataset('flights')
    df['month'] = df.month.map({'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'Jun':5, 'Jul':6, 'Aug':7, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}).astype(np.float32)
    df['passengers'] = df.passengers.astype(np.float32)
    df['year'] = df.year.astype(np.float32)
    return df
    
    

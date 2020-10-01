import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from ..kernel.helper import *
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_boston

def map_categories(df, column):
    df = df[column]
    try:
        s = sorted(df.unique())
    except:
        s = df.unique()
    d = { v:i for i, v in enumerate(s) }
    return df.map(d)

class TabularDatabunch:
    def __init__(self, train_X, train_y, valid_X, valid_y, batch_size=32, scale=True, dtype=np.float32, dtypey=None, scale_y=False, balance=False, shuffle=True, degree=1, num_workers=0, **kwargs):
        self.balanced = balance
        self.dtype = dtype
        self.dtypey = dtype if dtypey is None else dtypey
        if self.balanced:
            train_X, train_y = self.balance(train_X, train_y)
        self.create_transform_x(scale, train_X)
        train_X = self.transform_x(train_X)
        valid_X = self.transform_x(valid_X)
        self.create_transform_y(scale_y, train_y)
        train_y = self.transform_y(train_y)
        valid_y = self.transform_y(valid_y)
        if degree > 1:
            poly = PolynomialFeatures(degree, include_bias=False)
            train_X = poly.fit_transform(train_X)
            valid_X = poly.fit_transform(valid_X)
        self.train_ds = self.Xy(train_X, train_y)
        self.valid_ds = self.Xy(valid_X, valid_y)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.kwargs = kwargs

    def create_transform_y(self, scale_y, train_y):
        self.scale_y = scale_y
        if self.scale_y:
            if callable(scale_y):
                self._transform_y = scale_y
            else:
                t = StandardScaler()
                t.fit(train_y)
                self._transform_y = t.transform
                self._inverse_transform_y = t.inverse_transform

    def create_transform_x(self, scale, X):
        self.scale_x = scale
        if self.scale_x:
            X = X.astype(self.dtype)
            self._transform_x = {}
            self._inverse_transform_x = {}
            try:
                for c in scale:
                    break
            except:
                self.scale_x = list(range(X.shape[1]))
            for c in self.scale_x:
                t = StandardScaler()
                t.fit(X[:,c:c+1])
                self._transform_x[c] = t.transform
                self._inverse_transform_x[c] = t.inverse_transform
        else:
            self.scale_x = []

    def transform_y(self, y):
        if self.scale_y:
            y = self._transform_y(y)
        return y

    def inverse_transform_y(self, y):
        if self.scale_y:
            y = self._inverse_transform_y(to_numpy(y))
        return y

    def transform_x(self, X):
        if self.scale_x:
            X = X.astype(self.dtype)
            for c in self.scale_x:
                X[:, c:c+1] = self._transform_x[c](X[:, c:c+1])
        return X

    def inverse_transform_x(self, X):
        if self.scale_x:
            for c in self.scale_x:
                X[:, c:c+1] = self._inverse_transform_x[c](X[:, c:c+1])
        return X

    @staticmethod
    def balance(X, y):
        indices = [np.where(y==l)[0] for l in np.unique(y)]
        classlengths = [len(i) for i in indices]
        n = max(classlengths)
        mask = np.hstack([np.random.choice(i, n-l, replace=True) for l,i in zip(classlengths, indices)])
        indices = np.hstack([mask, range(len(y))])
        return X[indices], y[indices]

    def Xy(self, X, y):
        return TensorDataset(torch.from_numpy(X.astype(self.dtype)), torch.from_numpy(y.astype(self.dtypey)))

    @classmethod
    def from_list( cls, l, valid_perc=0.2, num_workers=0, random_state=None, dtype=np.float32, **kwargs ):
        t = np.matrix(l, dtype=dtype)
        if valid_perc > 0:
            train_X, valid_X, train_y, valid_y = train_test_split(t[:,:-1], t[:,-1], test_size=valid_perc, random_state=random_state)
        else:
            train_X = t[:,:-1]
            valid_X = train_X
            train_y = t[:, -1]
            valid_y = train_y
        return cls(train_X, train_y, valid_X, valid_y, num_workers=num_workers, **kwargs)

    @classmethod
    def from_pd( cls, df, target, *features, valid_perc=0.2, pin_memory=False, num_workers=0, random_state=None, **kwargs ):
        tensors = split_xy(df, target, *features, valid_perc=valid_perc)
        return cls(*tensors, pin_memory=pin_memory, num_workers=num_workers, **kwargs)

    @classmethod
    def from_np( cls, *arrays, pin_memory=False, num_workers=0, **kwargs ):
        return cls(*arrays, pin_memory=pin_memory, num_workers=num_workers, **kwargs)

    def reset(self):
        try:
            del self.valid_dl
        except: pass
        try:
            del self._train_dl
        except: pass

    def _dataloader(self, ds, shuffle=False):
        return DataLoader(ds, batch_size = self.batch_size, shuffle=shuffle, num_workers=self.num_workers, **self.kwargs)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = min(value, len(self.train_ds))
        self.reset()

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        self._num_workers = value
        self.reset()

    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            self._train_dl = self._dataloader(self.train_ds, shuffle=self.shuffle)
            return self._train_dl
   
    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            self._valid_dl = self._dataloader(self.valid_ds)
            return self._valid_dl

    @property
    def valid_tensors(self):
        return self.valid_ds.tensors

    @property
    def train_tensors(self):
        return self.train_ds.tensors

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
    def train_numpy(self):
        return to_numpy(self.train_X), to_numpy(self.train_y)

    @property
    def valid_numpy(self):
        return to_numpy(self.valid_X), to_numpy(self.valid_y)

    def sample(self, device=None):
        X, y = next(iter(self.train_dl))
        if device is not None:
            return X.to(device), y.to(device)
        return X, y

    def _plot(self, **kwargs):
        figure = plt.figure(**kwargs)
        return plt.axes()

    def plot_train(self, ind_x=None, **kwargs):
        p = self._plot(**kwargs)
        if ind_x is None:
            p.scatter(to_numpy(self.train_X), to_numpy(self.train_y))
        else:
            p.scatter(to_numpy(self.train_X[:,ind_x]), to_numpy(self.train_y))
        return p

    def plot_valid(**kwargs):
        p = self._plot(self, **kwargs)
        p.scatter(to_numpy(self.valid_X), to_numpy(self.valid_y))
        return p

    def scatter_train(self, x1=0, x2=1, s=1, figsize=(8,8), **kwargs):
        p = self._plot(figsize=figsize, **kwargs)
        x_0 = self.train_X[self.train_y == 0]
        x_1 = self.train_X[self.train_y == 1]
        p.scatter(to_numpy(x_0[:, x1]), to_numpy(x_0[:, x2]), marker='x', label=0, s=s)
        p.scatter(to_numpy(x_1[:, x1]), to_numpy(x_1[:, x2]), marker='+', label=1, s=s)
        p.legend()
        return p

def split_xy(df, target, *features, valid_perc=0.2):
    train, valid = train_test_split(df, test_size=valid_perc)
    if len(features) == 0:
        features = [c for c in df.columns if c != target]
    else:
        features = [c for c in features if c != target]
    a = [ds[c].to_numpy() for ds in [train, valid] for c in [features, target]]
    if len(a[0].shape) < 2:
        a[0] = a[0].reshape(a[0].shape[0], 1)
    if len(a[2].shape) < 2:
        a[2] = a[2].reshape(a[2].shape[0], 1)
    return a

def scaleindices(features, kwargs):
    if 'scale' in kwargs and kwargs['scale']:
        try:
            kwargs['scale'] = [ features.index(c) if isinstance(c, str) else c for c in kwargs['scale'] ]
        except: pass

def wines_pd():
    return pd.read_csv('/data/datasets/winequality-red.csv', delimiter=';', dtype=np.float32)

def wines(target, *features, valid_perc=0.2, threshold=None, **kwargs):
    scaleindices(features, kwargs)
    df = wines_pd()
    if threshold is not None:
        df['quality'] = ((df.quality >= threshold) * 1.0).astype(np.float32)
    arrays = split_xy(df, target, *features, valid_perc=valid_perc)
    return TabularDatabunch.from_np( *arrays, **kwargs )

def churn_df():
    return pd.read_csv('/data/datasets/telco_churn.csv')

def churn_dummies_df():
    df = churn_df().drop(columns='customerID')
    df['TotalCharges'] = np.where(df.TotalCharges.map(lambda x: len(x)) < 2, df.MonthlyCharges * df.tenure, df.TotalCharges)

    columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
    dfs = [df]
    for c in columns:
        if len(df[c].unique()) == 2:
            df[c] = df[c].astype('category').cat.codes
        else:
            d = pd.get_dummies(df[c], prefix=c)
            dfs.append( d[d.columns[:-1]] )
            df.drop(columns=[c], inplace=True)

    return pd.concat(dfs, axis=1).astype(np.float32)

def churn(*features,  **kwargs):
    scaleindices(features, kwargs)
    df = churn_dummies_df()
    return TabularDatabunch.from_pd(df, 'Churn', *features, **kwargs)

def movielens_df():
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    return pd.read_csv("/data/datasets/ml-1m/ratings.dat",sep='::', names=COLS, engine='python').drop(columns=['timestamp']).astype(int)

def movielens_movies_df():
    COLS = ['movie_id', 'title', 'genre']
    return pd.read_csv("/data/datasets/ml-1m/movies.dat",sep='::', names=COLS, engine='python')

def movielens(valid_perc=0.2, dtype=np.float32, **kwargs):
    df = movielens_df()
    df['rating'] = df.rating.astype(dtype)
    return TabularDatabunch.from_np( *split_xy(df, 'rating', 'user_id', 'movie_id', valid_perc=valid_perc), scale=False, **kwargs)

def dam(**kwargs):
    with open("/data/datasets/dam_water_data.pickle", "rb") as myfile:
        X_train, X_val, X_test, X_all, y_train, y_val, y_test, y_all = pickle.load(myfile)
    return TabularDatabunch.from_np(X_train.reshape(-1, 1), y_train, X_val.reshape(-1, 1), y_val, **kwargs)

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
    scaleindices(features, kwargs)
    df = boston_housing_prices_df()
    return TabularDatabunch.from_np( *split_xy(df, target, *features, valid_perc=valid_perc), **kwargs)

def banking_df():
    df = pd.read_csv("/data/datasets/bank-additional-full.csv", sep=';')
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
    scaleindices(features, kwargs)
    df = banking_cleaned()
    return TabularDatabunch.from_np( *split_xy(df, 'y', *features, valid_perc=valid_perc), **kwargs)

def rossmann_orig_df():
    dtypes = { 'Store': 'int', 'DayOfWeek': 'int', 'Sales': 'int', 'Customers': 'int', 'Open': 'int', 'Promo': 'int', 'SchoolHoliday': 'int', 'DayOfWeek': 'int', 'StateHoliday':'str', 'Date':'str'}
    df = pd.read_csv("/data/datasets/rossmann/train.csv", sep=',', dtype=dtypes, parse_dates=['Date'])
    return df

def rossmann_df():
    return rossmann_clean(rossmann_orig_df().drop(columns='Customers'))

def rossmann_clean(df):
    df['dayofweek_cat'] = map_categories(df, 'DayOfWeek')
    df['stateholiday_cat'] = map_categories(df, 'StateHoliday')
    df['Date'] = pd.to_datetime(df.Date)
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    df.drop(columns=['StateHoliday', 'Date'], inplace=True)
    return df

def rossmann(*features, valid_perc=0.2, dtype=np.long, dtypey=np.float32, scale=False, **kwargs):
    df = rossmann_df()
    return TabularDatabunch.from_np( *split_xy(df, 'Sales', *features, valid_perc=valid_perc), scale=scale, dtype=dtype, dtypey=dtypey, **kwargs)

def california_df():
    df = pd.read_csv("/data/datasets/california.csv", sep=',')
    return df

def california_cleaned_df(*features, valid_perc=0.2, **kwargs):
    df = california_df()
    df['near_bay'] = np.where(df.ocean_proximity == 'NEAR BAY', 1, 0)
    df['near_ocean'] = np.where(df.ocean_proximity == 'NEAR OCEAN', 1, 0)
    df['island'] = np.where(df.ocean_proximity == 'ISLAND', 1, 0)
    df['inland'] = np.where(df.ocean_proximity == 'INLAND', 1, 0)
    df.drop(columns=['ocean_proximity'], inplace=True)
    df = df.astype(np.float32)
    return df

def california(*features, valid_perc=0.2, **kwargs):
    scaleindices(features, kwargs)
    df = california_cleaned_df()
    return TabularDatabunch.from_np( *split_xy(df, 'median_house_value', *features, valid_perc=valid_perc), **kwargs)

def california_lookup(*features, scale=False, valid_perc=0.2, **kwargs):
    scaleindices(features, kwargs)
    df = california_df()
    df['near_bay'] = np.where(df.ocean_proximity == 'NEAR BAY', 1, 0)
    df['near_ocean'] = np.where(df.ocean_proximity == 'NEAR OCEAN', 1, 0)
    df['island'] = np.where(df.ocean_proximity == 'ISLAND', 1, 0)
    df['inland'] = np.where(df.ocean_proximity == 'INLAND', 1, 0)
    df['longitude_cat'] = map_categories(df, 'longitude')
    df['latitude_cat'] = map_categories(df, 'latitude')
    df['ocean_proximity_cat'] = map_categories(df, 'ocean_proximity')
    df['housing_median_age_cat'] = map_categories(df, 'housing_median_age')
    df = df.drop(columns='ocean_proximity').astype(np.float32)
    return TabularDatabunch.from_np( *split_xy(df, 'median_house_value', *features, valid_perc=valid_perc), scale=scale, **kwargs)

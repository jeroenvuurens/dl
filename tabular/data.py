import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from ..kernel.helper import *
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import load_boston

def map_categories(df, column):
    df = df[column]
    try:
        s = sorted(df.unique())
    except:
        s = df.unique()
    d = { v:i for i, v in enumerate(s) }
    df = df.map(d).astype(np.long)
    return df

def scale(arr):
    if arr.dtype.kind == 'f':
        scaler = StandardScaler()
        scaler.fit(arr)
        return scaler

class ArrayDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *arrays):
        assert all(arrays[0].shape[0] == a.shape[0] for a in arrays)
        self.tensors = [ torch.from_numpy(a) for a in arrays ]
        super().__init__(*self.tensors)

    def inverse_transform_y(self, y):
        return y

    def inverse_transform_x(self, *x):
        return x

    def inverse_transform(self, *sample):
        return sample

class ScaledDataset(ArrayDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *arrays):
        self.scalers = tuple( self._scaler(a) for a in arrays )
        arrays = [ self._fit(a, s) for a, s in zip(arrays, self.scalers) ]
        super().__init__(*arrays)

    def _scaler(self, arr):
        if arr.dtype.kind == 'f':
            min = arr.min()
            max = arr.max()
            if max - min < 0.1 or min < -2 or max > 2:
                scaler = StandardScaler()
                if len(arr.shape) == 1:
                    arr = arr.reshape(-1, 1)
                scaler.fit(arr)
                return scaler

    def _fit(self, arr, scaler):
        if scaler is not None:
            if len(arr.shape) == 1:
                arr = arr.reshape(-1, 1)
                arr = scaler.transform(arr)
                arr = arr.reshape(-1)
            else:
                arr = scaler.transform(arr)
        return arr

    def inverse_transform_y(self, y):
        #try:
            if self.scalers[-1] is not None:
                y = to_numpy(y)
                if len(y.shape) == 1:
                    y = y.reshape(-1, 1)
                    y = torch.tensor(self.scalers[-1].inverse_transform(y))
                    y = y.reshape(-1)
                else:
                    y = torch.tensor(self.scalers[-1].inverse_transform(y))
        #except:
            return y

    def inverse_transform_x(self, *x):
        return [ x[i] if self.scalers[i] is None else torch.tensor(self.scalers[i].inverse_transform(to_numpy(x[i]))) for i in range(len(x)) ]

    def inverse_transform(self, sample):
        s = self.inverse_transform_x(sample[:-1])
        s.append( self.inverse_transform_y( sample[-1] ) )
        return s

    def to_numpy(self):
        return ( to_numpy(t) for t in self.tensors )

class TabularDatabunch:
    def __init__(self, ds, train_indices, valid_indices, batch_size=32, num_workers=0):
        self.ds = ds
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            self._train_dl = self._dl(self.train_indices, self.batch_size, self.num_workers)
            return self._train_dl

    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            self._valid_dl = self._dl(self.valid_indices, self.batch_size, self.num_workers)
            return self._valid_dl

    def _dl(self, indices, batch_size, num_workers):
        sampler = SubsetRandomSampler(indices)
        return DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler)

    @property
    def train_numpy(self):
        return tuple( a[self.train_indices] for a in self.ds.to_numpy() )

    @property
    def valid_numpy(self):
        return tuple( a[self.valid_indices] for a in self.ds.to_numpy() )

    def inverse_transform_y(self, y):
        return self.ds.inverse_transform_y(y)

    def inverse_transform(self, *arrays):
        return self.ds.inverse_transform(*arrays)

    def inverse_transform_x(self, *x):
        return self.ds.inverse_transform_x(*x)

    @staticmethod
    def split(arrays, valid_perc):
        dataset_size = len(arrays[0])
        indices = np.array(range(dataset_size))
        split = int(np.floor(valid_perc * dataset_size))
        np.random.shuffle(indices)
        return indices[split:], indices[:split]

    @staticmethod
    def balance_y(y, train_indices):
        y = y[train_indices]
        indices = [np.where(y==l)[0] for l in np.unique(y)]
        classlengths = [len(i) for i in indices]
        n = max(classlengths)
        mask = np.hstack([np.random.choice(i, n-l, replace=True) for l,i in zip(classlengths, indices)])
        indices = np.hstack([mask, range(len(y))])
        return train_indices[indices]

    @staticmethod
    def polynomials(arrays, degree):
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
    def from_arrays(cls, *arrays, batch_size=32, scale=True, balance=False, degree=1, num_workers=0, random_seed=0, valid_perc=0.2):
        np.random.seed(random_seed)
        if degree > 1:
            arrays = cls.polynomials(arrays, degree)
        train_indices, valid_indices = cls.split(arrays, valid_perc)
        if balance:
             train_indices = cls.balance_y(arrays[-1], train_indices)
        if scale:
            ds = ScaledDataset( *arrays )
        else:
            ds = ArrayDataset( *arrays )
        return TabularDatabunch(ds, train_indices, valid_indices, batch_size=batch_size, num_workers=num_workers)

    @classmethod
    def from_list( cls, l, num_workers=0, random_state=None, dtype=np.float32, batch_size=1 ):
        """ Mostly used for demonstrating binary operators, therefore, scale 
        and balance are turned off by default
        """
        t = np.array(l, dtype=dtype)
        #tensors = t[:, :-1], t[:, -1].
        ds = TensorDataset( torch.tensor(t[:, :-1]), torch.tensor(t[:, -1]).view(-1) )
        ds.inverse_transform_y = lambda x: x
        ds.inverse_transform_x = lambda *x: x
        ds.inverse_transform = lambda *x: x
        indices = list(range(len(l)))
        return TabularDatabunch(ds, indices, indices, batch_size=batch_size, num_workers=num_workers)

    @classmethod
    def from_df( cls, df, target, *features, batch_size=64, **kwargs ):
        features = expand_features(df, target, *features)
        arrays = cls.df_split_xy(df, target, *features)
        return cls.from_arrays(*arrays, batch_size=batch_size, **kwargs)

    @staticmethod
    def df_split_xy2(df, target, *features):
        if not isinstance(target, (list, tuple)):
            target = [ target ]
        return df[[ f for f in features]].to_numpy(), df[target].to_numpy()

    @staticmethod
    def df_split_xy(df, target, *features):
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
    def train_X(self, index=0):
        return self.ds.tensors[index][self.train_indices]

    @property
    def train_y(self):
        return self.ds.tensors[-1][self.train_indices]

    @property
    def valid_X(self):
        return self.ds.tensors[index][self.valid_indices]

    @property
    def valid_y(self):
        return self.ds.tensors[-1][self.valid_indices]

    @property
    def train_X_orig(self):
        return self.inverse_transform_x(self.train_X)[0]

    @property
    def train_y_orig(self):
        return self.inverse_transform_y(self.train_y)

    @property
    def valid_X_orig(self):
        return self.inverse_transform_x(self.valid_X)[0]

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
    if not isinstance(target, (list, tuple)):
        target = [ target ]
    if len(features) == 0:
        return [c for c in df.columns if c not in target]
    else:
        return [c for c in features if c not in target]

def wines_pd():
    return pd.read_csv('/data/datasets/winequality-red.csv', delimiter=';', dtype=np.float32)

def wines(target, *features, valid_perc=0.2, threshold=None, **kwargs):
    df = wines_pd()
    if threshold is not None:
        df[target] = ((df[target] >= threshold) * 1.0).astype(np.float32)
    features = expand_features(df, target, *features)
    return TabularDatabunch.from_df( df, target, *features, valid_perc=valid_perc, **kwargs )

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
    df = churn_dummies_df()
    return TabularDatabunch.from_df(df, 'Churn', *features, **kwargs)

def movielens_df():
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv("/data/datasets/ml-1m/ratings.dat",sep='::', names=COLS, engine='python').astype(np.long)
    df.timestamp = pd.to_datetime(df.timestamp, unit='s')
    df['year'] = df.timestamp.dt.year
    return df

def movielens_movies_df():
    COLS = ['movie_id', 'title', 'genre']
    return pd.read_csv("/data/datasets/ml-1m/movies.dat",sep='::', names=COLS, engine='python')

def movielens(*features, valid_perc=0.2, dtype=np.float32, **kwargs):
    if len(features) == 0:
        features = ('user_id', 'movie_id')
    df = movielens_df()
    df['rating'] = df.rating.astype(dtype)
    return TabularDatabunch.from_df( df, 'rating', *features, valid_perc=valid_perc, scale=False, **kwargs)

def dam(**kwargs):
    with open("/data/datasets/dam_water_data.pickle", "rb") as myfile:
        X_train, X_val, X_test, X_all, y_train, y_val, y_test, y_all = pickle.load(myfile)
    X_all = X_all.reshape(-1, 1).astype(np.float32)
    y_all = y_all.astype(np.float32)
    return TabularDatabunch.from_arrays(X_all, y_all, **kwargs)

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
    features = expand_features(df, target, *features)
    return TabularDatabunch.from_df( df, target, *features, valid_perc=valid_perc, **kwargs)

def banking_df():
    df = pd.read_csv("/data/datasets/bank-additional-full.csv", sep=';')
    return df

def banking_cleaned():
    df = banking_df()
    df['y'] = np.where(df.y == 'yes', 1.0, 0.0).astype(np.float32)
    columns = ['marital', 'job', 'poutcome', 'contact', 'housing', 'loan', 'default', 'education', 'day_of_week', 'month']
    for c in columns:
        df[f'{c}_ordinal'] = df[c].astype('category').cat.codes
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

def banking_desc():
    with open('/data/datasets/bank-additional-names.txt', 'r') as fin:
        print(''.join(fin.readlines()))

def banking(*features, valid_perc=0.2, dtype_y=np.float32, **kwargs):
    df = banking_cleaned()
    df['y'] = df.y.astype(dtype_y)
    features = expand_features(df, 'y', *features)
    return TabularDatabunch.from_df( df, 'y', *features, valid_perc=valid_perc, **kwargs)

def rossmann_orig_df():
    dtypes = { 'Store': 'int', 'DayOfWeek': 'int', 'Sales': 'int', 'Customers': 'int', 'Open': 'int', 'Promo': 'int', 'SchoolHoliday': 'int', 'DayOfWeek': 'int', 'StateHoliday':'str', 'Date':'str'}
    df = pd.read_csv("/data/datasets/rossmann/train.csv", sep=',', dtype=dtypes, parse_dates=['Date'])
    df = df[ df.Sales > 0 ]
    return df

def rossmann_df():
    return rossmann_clean(rossmann_orig_df().drop(columns='Customers'))

def rossmann_clean(df):
    df['Date'] = pd.to_datetime(df.Date)
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    df['dayofweek_cat'] = map_categories(df, 'DayOfWeek')
    df['stateholiday_cat'] = map_categories(df, 'StateHoliday')
    df['year_cat'] = map_categories(df, 'year')
    df['month_cat'] = map_categories(df, 'month')
    df.drop(columns=['StateHoliday', 'Date'], inplace=True)
    return df

def rossmann(*features, valid_perc=0.2, scale=False, **kwargs):
    df = rossmann_df()
    features = expand_features(df, 'Sales', *features)
    return TabularDatabunch.from_np( df, 'Sales', *features, valid_perc=valid_perc, scale=scale, **kwargs)

def california_df():
    df = pd.read_csv("/data/datasets/california.csv", sep=',')
    return df

def california_cleaned_df():
    df = california_df()
    df['near_bay'] = np.where(df.ocean_proximity == 'NEAR BAY', 1, 0)
    df['near_ocean'] = np.where(df.ocean_proximity == 'NEAR OCEAN', 1, 0)
    df['island'] = np.where(df.ocean_proximity == 'ISLAND', 1, 0)
    df['inland'] = np.where(df.ocean_proximity == 'INLAND', 1, 0)
    df.drop(columns=['ocean_proximity'], inplace=True)
    df = df.astype(np.float32)
    return df

def california(*features, valid_perc=0.2, **kwargs):
    df = california_cleaned_df()
    features = expand_features(df, 'median_house_values', *features)
    return TabularDatabunch.from_df( df, 'median_house_value', *features, valid_perc=valid_perc, **kwargs)

def california_lookup_df(*features, scale=False, valid_perc=0.2, **kwargs):
    df = california_df().astype(np.float32, errors='ignore')
    df['near_bay'] = np.where(df.ocean_proximity == 'NEAR BAY', 1, 0)
    df['near_ocean'] = np.where(df.ocean_proximity == 'NEAR OCEAN', 1, 0)
    df['island'] = np.where(df.ocean_proximity == 'ISLAND', 1, 0)
    df['inland'] = np.where(df.ocean_proximity == 'INLAND', 1, 0)
    df['longitude_cat'] = map_categories(df, 'longitude')
    df['latitude_cat'] = map_categories(df, 'latitude')
    df['ocean_proximity_cat'] = map_categories(df, 'ocean_proximity')
    df['housing_median_age_cat'] = map_categories(df, 'housing_median_age')
    df = df.drop(columns='ocean_proximity')
    return df

def california_lookup(*features, scale=False, valid_perc=0.2, **kwargs):
    df = california_lookup_df()
    features = expand_features(df, 'median_house_value', *features)
    return TabularDatabunch.from_df( df, 'median_house_value', *features, valid_perc=valid_perc, scale=scale, **kwargs)

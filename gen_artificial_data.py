import sklearn
import pandas as pd
import numpy as np
import itertools

from sklearn.datasets import make_classification

from imbalanced import *
from imbalanced import _split_minor_major

def generate_data(min_maj_rate, n_train, n_features, n_informative, n_clusters, class_sep, random_state, n_test = 0):
    if n_clusters > 2 ** (n_informative - 1) or n_informative > n_features:
        return None, None, None, None
    X, y = sklearn.datasets.make_classification(
        n_samples=n_train + n_test, n_features=n_features, n_informative=n_informative, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=n_clusters, weights=[1.0 / (1 + min_maj_rate), min_maj_rate / (1 + min_maj_rate)], 
        flip_y=0.01, class_sep=class_sep, hypercube=False, shift=0.0, scale=1.0, shuffle=True, random_state=random_state)
    X_train = X[:n_train,:]
    y_train = y[:n_train]
    X_test = X[n_train:,:]
    y_test = y[n_train:]
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    N_start = 600
    N_end = 1000
    # Fix order of params by introducing this list 
    # (in dict, order of keys may be different every time we use it - this can cause problems with random state setting)
    params_names = ['min_maj_rate', 'n_train', 'n_features', 'n_clusters', 'class_sep', 'random_state']
    params_possible_values = {
        'min_maj_rate': np.arange(0.05, 0.36, 0.01).tolist(),
        'n_train': np.arange(200, 1000, 50).tolist(),
        'n_features': np.arange(6, 40, 2).tolist(),
        'n_clusters': [1, 2, 3, 4, 5],
        'class_sep': np.arange(0.85, 1.16, 0.01).tolist(),
        'random_state': range(100)
    }
    params_possible_n = {param: len(params_possible_values[param]) for param in params_names}

    df_gen_info = pd.DataFrame(columns=params_names, index=range(N_start, N_end))
    for i in range(N_start, N_end):
        print i
        np.random.seed(i)
        d = {} # dict with values of params for current artificial dataset
        for param in params_names:
            d[param] = params_possible_values[param][np.random.randint(params_possible_n[param])]
        #print d
        df_gen_info.loc[i] = [d[param] for param in params_names]
        try:
            X, y, Xt, yt = generate_data(d['min_maj_rate'], d['n_train'], d['n_features'], d['n_features'], d['n_clusters'], d['class_sep'], d['random_state'], n_test=0)
            df = pd.DataFrame(X, columns=range(X.shape[1]))
            df['label'] = y
            df.to_csv('./data_csv/data_art_%03d.csv' % i)
        except ValueError:
            print 'ValueError occured, skip this set of parameters...'
            pass
    df_gen_info.to_csv('./gen_artificial_info.csv')
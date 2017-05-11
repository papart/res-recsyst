import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import copy
import scipy

# Common helper functions
def print_full(x, rows=None):
    if type(rows) is type(1):
        pd.set_option('display.max_rows', rows)
    else:
        pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    
def ttest_onetail(a, b, axis=0, equal_var=False, alternative='greater'):
    t_raw, p_raw = scipy.stats.ttest_ind(a, b, axis=axis, equal_var=equal_var)
    
    if np.isnan(t_raw):
        return 1.0
    if alternative in ['greater', '>']:
        if t_raw > 0:
            p = 0.5 * p_raw
        else:
            p = 1 - 0.5 * p_raw 
    elif alternative in ['smaller', '<']:
        if t_raw < 0:
            p = 0.5 * p_raw
        else:
            p = 1 - 0.5 * p_raw
    else:
        print 'Unknown parameter: alternative="%s"' % alternative
    return p

# Functions for quality measurement
def av_q_gain_compto_beststatic(Q, modes_pred):
    beststatic_av_quality = Q.sum().max() / len(Q)
    recsystem_av_quality = sum([Q[modes_pred[i]].values[i] for i in range(len(Q))]) / len(Q)
    #print recsystem_av_quality - Q.sum() / len(Q)
    return recsystem_av_quality - beststatic_av_quality

def q_metric_ara(Q, modes_pred):
    # in case with 2 possible preprocessing modes (e.g. resample algorithms)
    # the same is accuracy of recomendation system
    best_q = Q.max(axis = 1).values
    worst_q = Q.min(axis = 1).values
    recsystem_q = np.array([Q[modes_pred[i]].values[i] for i in range(len(Q))])
    return np.nan_to_num((recsystem_q - worst_q) / (best_q - worst_q)).mean()
    
def av_q_gain_compto_ideal(Q, modes_pred):
    best_q = Q.max(axis = 1).values
    recsystem_q = np.array([Q[modes_pred[i]].values[i] for i in range(len(Q))])
    return (recsystem_q - best_q).mean()

def av_relative_q_gain_compto_ideal(Q, modes_pred):
    best_q = Q.max(axis = 1).values
    recsystem_q = np.array([Q[modes_pred[i]].values[i] for i in range(len(Q))])
    return ((recsystem_q - best_q) / best_q).mean() 

# Helper functions for prediction
def predict_proba_n_folds(clf, X, y, sample_weight = None, n_folds = 10):
    #X - pd.DataFrame, y - pd.Series
    q_iterator = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=1)
    probas_predict = np.zeros(len(y))
    for train_idx, test_idx in q_iterator:
        #Get train and test subdatasets
        X_train = X.values[train_idx, :]
        if sample_weight is None:
            sw_train = None
        else:
            sw_train = sample_weight[train_idx]
        y_train = y.values[train_idx]
        X_test = X.values[test_idx, :]
        y_test = y.values[test_idx]

        #Transform data
        scl = StandardScaler()
        scl.fit(X_train)
        X_train = scl.transform(X_train)
        X_test = scl.transform(X_test)
        if sample_weight is None:
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train, sample_weight = sw_train)
        probas_predict[test_idx] = clf.predict_proba(X_test)[:,1]
    return probas_predict

def regression_n_folds(rgr, X, y, n_folds = 10):
    q_iterator = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=1)
    prediction = np.zeros(len(y))
    
    for train_idx, test_idx in q_iterator:
        #Get train and test subdatasets
        X_train = X.values[train_idx, :]
        y_train = y.values[train_idx]
        X_test = X.values[test_idx, :]
        y_test = y.values[test_idx]

        #Transform data
        scl = StandardScaler()
        scl.fit(X_train)
        X_train = scl.transform(X_train)
        X_test = scl.transform(X_test)

        rgr.fit(X_train, y_train)
        prediction[test_idx] = rgr.predict(X_test)
    return prediction

def quantize_rgr_prediction(y_pred, low=1.25, high=10.0, step = 0.25):
    y_result = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        val = 0.25 * int(np.round(y_pred[i]/step))
        if val <= low:
            y_result[i] = low
        elif val >= high:
            y_result[i] = high
        else:
            y_result[i] = val
    return y_result

def quantize(val, l):
    distances = abs(np.array(l) - val)
    return l[np.argmin(distances)]

class ModClassifier():
    # Class for classification models. 
    # Supports classification for one-class datasets
    clf = None
    one_class_label = None
    labels = None
    
    def __init__(self, base_clf):
        self.clf = copy.deepcopy(base_clf)
    
    def fit(self, X, y, labels=[0,1]):
        if len(np.unique(y)) == 1:
            # One-class dataset. 
            self.one_class_label = np.array(y)[0]
            self.labels = np.array(labels)
        else:
            # Regular dataset (assume that it consists of two classes)
            self.one_class_label = None
            self.clf.fit(X, y)
    
    def predict(self, X):
        if self.one_class_label is not None:
            # One-class case
            return np.array([self.one_class_label] * len(X))
        else:
            # Regular case
            return self.clf.predict(X)
    def predict_proba(self, X):
        if self.one_class_label is not None:
            # One-class case
            idx = np.where(self.labels == self.one_class_label)[0][0]
            probas = np.zeros((len(X), 2))
            probas[:,idx] = 1.0
            return probas
        else:
            # Regular case
            return self.clf.predict_proba(X)

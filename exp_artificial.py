import sklearn
import pandas as pd
import numpy as np
import itertools

from sklearn.datasets import make_classification
from sklearn import metrics
from imbalanced import *
from imbalanced import _split_minor_major

from enum import Enum
import matplotlib.pyplot as plt

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class ScoreMetric(Enum):
    accuracy = {"pred_type": "label",
               "function": metrics.accuracy_score}
    #f1 = {"pred_type": "label",
    #      "function": metrics.f1_score}
    #precision = {"pred_type": "label",
    #             "function": metrics.precision_score}
    #recall = {"pred_type": "label",
    #          "function": metrics.recall_score}
    prc_auc = {"pred_type": "probability",
               "function": metrics.average_precision_score}
    roc_auc = {"pred_type": "probability",
               "function": metrics.roc_auc_score}

classifiers = {
'logreg_l1': {"classifier": LogisticRegression(penalty='l1'), "param_grid": {"C": 10.0 ** np.arange(-3, 4)}},
'logreg_l2': {"classifier": LogisticRegression(penalty='l2'), "param_grid": {"C": 10.0 ** np.arange(-3, 4)}},
'svm_rbf': {"classifier": SVC(kernel='rbf', probability=True), "param_grid": {"C": 10.0 ** np.arange(-3, 4), "gamma": 10.0 ** np.arange(-3, 4)}},
'svm_lin': {"classifier": SVC(kernel='linear', probability=True), "param_grid": {"C": 10.0 ** np.arange(-3, 4)}},
#'svm_poly': {"classifier": SVC(kernel='poly', probability=True), "param_grid": {"C": 10.0 ** np.arange(-3, 4), "gamma": 10.0 ** np.arange(-3, 4), "degree": [2, 3, 4]}},
'knn': {"classifier": KNeighborsClassifier(), "param_grid": {"n_neighbors": range(1,6)}},
'dtree': {"classifier": DecisionTreeClassifier(), "param_grid": {}}
}

class ResamplingEnum(Enum):
    bootstrap = {'oversampling': True, 'function': Bootstrap}
    smote =     {'oversampling': True, 'function': SMOTE}
    rus =       {'oversampling': False, 'function': RUS}
    #sinop =     {'oversampling': True, 'function': SINOP}
    #nothing = None

def resample(X, y, final_min_maj_rate, strategy, **kwargs):
    if strategy == 'nothing':
        return X, y

    X_min, y_min, X_maj, y_maj = _split_minor_major(X, y)
    if ResamplingEnum[strategy].value['oversampling']:
        n_min_final = int(np.floor(len(y_maj) * final_min_maj_rate))
        n_samples = n_min_final - len(y_min)
        X_min_final = ResamplingEnum[strategy].value['function'](X_min, n_samples, **kwargs)
        y_min_final = np.array([y_min[0]] * n_min_final)
        X_final = np.vstack((X_min_final, X_maj))
        y_final = np.concatenate((y_min_final, y_maj))
    else:
        n_maj_final = int(np.floor(len(y_min) / final_min_maj_rate))
        n_samples = len(y_maj) - n_maj_final
        X_maj_final = ResamplingEnum[strategy].value['function'](X_maj, n_samples, **kwargs)
        y_maj_final = np.array([y_maj[0]] * n_maj_final)
        X_final = np.vstack((X_min, X_maj_final))
        y_final = np.concatenate((y_min, y_maj_final))
    return X_final, y_final       


def score(clf, X, y, metric_name = "prc_auc"):
    if metric_name not in ScoreMetric._member_names_:
        raise ValueError("'%s' is not a valid score metric" % metric_name)
    pred_type = ScoreMetric[metric_name].value['pred_type']
    metric_function = ScoreMetric[metric_name].value['function']
    if pred_type == 'label':
        y_pred = clf.predict(X)
        return metric_function(y, y_pred)
    elif pred_type == 'probability':
        proba_pred = clf.predict_proba(X)[:, 1]
        return metric_function(y, proba_pred)

def draw_pr_curve(clf, X, y):
    proba_pred = clf.predict_proba(X)[:, 1]
    precision, recall, thrs = metrics.precision_recall_curve(y, proba_pred, pos_label=1)
    prc_auc = score(X, y, clf, "prc_auc")
    plt.clf()
    plt.plot(recall, precision, label='PR-curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve (area = %0.3f)' % prc_auc)
    plt.legend(loc="upper right")
    plt.show()

def generate_data(min_maj_rate, n_train, n_features, n_informative, n_clusters, class_sep, random_state, n_test = 10000):
    if n_clusters > 2 ** (n_informative - 1) or n_informative > n_features:
        return None, None, None, None
    X, y = sklearn.datasets.make_classification(
        n_samples=n_train + n_test, n_features=n_features, n_informative=n_informative, n_redundant=0, n_repeated=0, 
        n_classes=2, n_clusters_per_class=n_clusters, weights=[1.0 / (1 + min_maj_rate), min_maj_rate / (1 + min_maj_rate)], 
        flip_y=0.0, class_sep=class_sep, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=random_state)
    X_train = X[:n_train,:]
    y_train = y[:n_train]
    X_test = X[n_train:,:]
    y_test = y[n_train:]
    return X_train, y_train, X_test, y_test

def exp_once(X_train, y_train, X_test, y_test, resampling_name, final_min_maj_rate, results_1exp):
    results_1exp['resample_strategy'] = resampling_name
    results_1exp['final_min_maj_rate'] = final_min_maj_rate

    X_train_res, y_train_res = resample(X_train, y_train, final_min_maj_rate, resampling_name)
    for clf_name in classifiers.keys():
        #print clf_name
        # Get classifier by its name
        clf = classifiers[clf_name]['classifier']
        param_grid = classifiers[clf_name]['param_grid']
        
        # Find best params for classifier using CV
        cv_iterator = StratifiedKFold(y_train_res, n_folds=min(5, sum(y_train_res)), shuffle=True)
        gscv = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, scoring=score,
            refit=False, cv=cv_iterator, n_jobs = 1, n_iter = 5)
        #gscv = GridSearchCV(estimator=base_clf, param_grid=param_grid, scoring=score, refit=False, cv=cv_iterator, n_jobs = 4)
        
        # Fit and test classifier with best params
        gscv.fit(X_train_res, y_train_res)
        clf.set_params(**gscv.best_params_)
        clf.fit(X_train_res, y_train_res)

        for metric_name in ScoreMetric._member_names_:
            col_name = [clf_name + '_' + metric_name]
            col_value = score(clf, X_test, y_test, metric_name)
            results_1exp[col_name] = col_value

    return results_1exp

if __name__ == '__main__':
    min_maj_rate_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] #important!
    n_train_list = [200]#, 200, 500, 1000]
    n_features_list = [30]#[30, 20, 10, 5, 2]#[5, 10, 20, 30]
    n_clusters_list = [1, 2, 3]
    class_sep_list = [0.9, 1.0, 1.1]
    n_inf_coef_list = [1.0]
    random_state_list = range(5) # Artem: range(5), Pavel: range(5,10)

    resample_multiplier_list = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    #final_min_maj_rate_list = [0.75, 1.0, 1.25, 1.5]
    
    columns_data = ['min_maj_rate', 'n_train', 'min_maj_rate_train', 'n_features', 'n_informative', 'n_clusters', 'class_sep', 'random_state']
    columns_resample = ['resample_strategy', 'final_min_maj_rate']
    columns_score = [i + '_' + j for i, j in itertools.product(classifiers.keys(), ScoreMetric._member_names_)]
    blank_columns = columns_data + columns_score + columns_resample
    results_blank = pd.DataFrame([np.zeros(len(blank_columns))], columns=blank_columns)
    results = results_blank.copy()

    product_obj = itertools.product(
        min_maj_rate_list, n_train_list, n_features_list, n_inf_coef_list, n_clusters_list, class_sep_list, random_state_list)
    
    i_exp = 1
    n_experiments = len(min_maj_rate_list) * len(n_train_list) * len(n_features_list) * len(n_clusters_list)\
        * len(class_sep_list) * len(random_state_list)
    for min_maj_rate, n_train, n_features, n_inf_coef, n_clusters, class_sep, random_state in product_obj:
        results.to_csv('exp_results_f30_200.csv')
        print "  Dataset #%d out of %d" % (i_exp, n_experiments)
        print min_maj_rate, n_train, n_features, n_inf_coef, n_clusters, class_sep, random_state
        #results.to_csv('exp_results_noresample_f30.csv')
        # Generate data
        n_informative = int(n_features * n_inf_coef)
        #print min_maj_rate, n_train, n_features, n_informative, n_clusters, class_sep, random_state
        X_train, y_train, X_test, y_test = generate_data(min_maj_rate, 
            n_train, n_features, n_informative, n_clusters, class_sep, random_state)
        if X_train is None or sum(y_train) < 2 or n_features >= 2*n_train:
            print 'Dataset with bad properties'
            i_exp += 1
            continue
        min_maj_rate_train = float(sum(y_train)) / (len(y_train) - sum(y_train))
        min_maj_rate_test = float(sum(y_test)) / (len(y_test) - sum(y_test))
        print '  Train sample. Total: %d, min/maj: %.3f' % (len(y_train), min_maj_rate_train)
        print '  Test sample. Total: %d, min/maj: %.3f' % (len(y_test), min_maj_rate_test)

        # Standardize data
        scl = StandardScaler()
        scl.fit(X_train)
        X_train = scl.transform(X_train)
        X_test = scl.transform(X_test)

        #X_train_res, y_train_res = resample('rus', X_train, y_train, 0.6)
        
        # Create blank for results of experiment
        results_1exp = results_blank.copy()
        results_1exp['min_maj_rate'] = min_maj_rate
        results_1exp['n_train'] = n_train
        results_1exp['min_maj_rate_train'] = min_maj_rate_train
        results_1exp['n_features'] = n_features
        results_1exp['n_informative'] = n_informative
        results_1exp['n_clusters'] = n_clusters
        results_1exp['class_sep'] = class_sep
        results_1exp['random_state'] = random_state
        
        
        for resampling_name in ResamplingEnum._member_names_:
            print 'Resampling: %s' % resampling_name
            if resampling_name == 'nothing':
                results_add = exp_once(X_train, y_train, X_test, y_test, resampling_name, 1.0, results_1exp)
                results = pd.concat([results, results_add], axis = 0)
            else:
                for resample_multiplier in resample_multiplier_list:
                    final_min_maj_rate = min_maj_rate_train * resample_multiplier
                    print 'Final minor/major rate: %.3f' % final_min_maj_rate
                    results_add = exp_once(X_train, y_train, X_test, y_test, resampling_name, final_min_maj_rate, results_1exp)
                    results = pd.concat([results, results_add], axis = 0)
        i_exp += 1
    
    

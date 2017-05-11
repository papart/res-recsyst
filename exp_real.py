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
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sys

# Constant used as random state eveywhere
RANDOM_STATE = 1

class ScoreMetric(Enum):
    #accuracy = {"pred_type": "label",
    #            "function": metrics.accuracy_score}
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
'logreg_l1': {"classifier": LogisticRegression(penalty='l1', random_state=RANDOM_STATE), "param_grid": {"C": 10.0 ** np.arange(-3, 4)}},
'logreg_l2': {"classifier": LogisticRegression(penalty='l2', random_state=RANDOM_STATE), "param_grid": {"C": 10.0 ** np.arange(-3, 4)}},
#'svm_rbf': {"classifier": SVC(kernel='rbf', probability=True), "param_grid": {"C": 10.0 ** np.arange(0, 1), "gamma": [0]}},#10.0 ** np.arange(0, 1)}},
#'svm_lin': {"classifier": SVC(kernel='linear', probability=True), "param_grid": {"C": 10.0 ** np.arange(-3, 4)}},
#'svm_poly': {"classifier": SVC(kernel='poly', probability=True), "param_grid": {"C": 10.0 ** np.arange(-3, 4), "gamma": 10.0 ** np.arange(-3, 4), "degree": [2, 3, 4]}},
'knn': {"classifier": KNeighborsClassifier(), "param_grid": {"n_neighbors": range(1,6)}},
'dtree': {"classifier": DecisionTreeClassifier(random_state=RANDOM_STATE), "param_grid": {}}
}

# Define functions for performing SMOTE with particular value of k
# Also define functions for perfoming resampling with fixed random_state
def SMOTE_1_rs(X, n_samples):
    return SMOTE(X, n_samples, k=1, random_state=RANDOM_STATE)

def SMOTE_3_rs(X, n_samples):
    return SMOTE(X, n_samples, k=3, random_state=RANDOM_STATE)

def SMOTE_5_rs(X, n_samples):
    return SMOTE(X, n_samples, k=5, random_state=RANDOM_STATE)

def SMOTE_7_rs(X, n_samples):
    return SMOTE(X, n_samples, k=7, random_state=RANDOM_STATE)

def Bootstrap_rs(X, n_samples):
    return Bootstrap(X, n_samples, random_state=RANDOM_STATE)

def RUS_rs(X, n_samples):
    return RUS(X, n_samples, random_state=RANDOM_STATE)



class ResamplingEnum(Enum):
    bootstrap = {'oversampling': True, 'function': Bootstrap_rs}
    #smote =     {'oversampling': True, 'function': SMOTE}
    rus =       {'oversampling': False, 'function': RUS_rs}
    smote1 = {'oversampling': True, 'function': SMOTE_1_rs}
    smote3 = {'oversampling': True, 'function': SMOTE_3_rs}
    smote5 = {'oversampling': True, 'function': SMOTE_5_rs}
    smote7 = {'oversampling': True, 'function': SMOTE_7_rs}
    #sinop =     {'oversampling': True, 'function': SINOP}
    nothing = None

def resample(X, y, min_maj_rate_final, strategy, **kwargs):
    """
    Applies resampling to objects X using labels y
    """
    if strategy == 'nothing':
        return X, y

    X_min, y_min, X_maj, y_maj = _split_minor_major(X, y)
    if ResamplingEnum[strategy].value['oversampling']:
        n_min_final = int(np.floor(len(y_maj) * min_maj_rate_final))
        n_samples = n_min_final - len(y_min)
        X_min_final = ResamplingEnum[strategy].value['function'](X_min, n_samples, **kwargs)
        y_min_final = np.array([y_min[0]] * n_min_final)
        X_final = np.vstack((X_min_final, X_maj))
        y_final = np.concatenate((y_min_final, y_maj))
    else:
        n_maj_final = int(np.floor(len(y_min) / min_maj_rate_final))
        n_samples = len(y_maj) - n_maj_final
        X_maj_final = ResamplingEnum[strategy].value['function'](X_maj, n_samples, **kwargs)
        y_maj_final = np.array([y_maj[0]] * n_maj_final)
        X_final = np.vstack((X_min, X_maj_final))
        y_final = np.concatenate((y_min, y_maj_final))
    return X_final, y_final       


def score(clf, X, y, metric_name = "prc_auc"):
    """
    Calculate score metric of classifier clf on dataset (X, y).
    """
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

def exp_once(X_train, y_train, X_test, y_test, resampling_name, resample_multiplier, results_1exp, n_folds=5, n_iter=5):
    min_maj_rate_final = min_maj_rate_train * resample_multiplier
    results_1exp['resample_strategy'] = resampling_name
    results_1exp['resample_multiplier'] = resample_multiplier
    results_1exp['min_maj_ratio_final'] = min_maj_rate_final

    X_train_res, y_train_res = resample(X_train, y_train, min_maj_rate_final, resampling_name)
    for clf_name in classifiers.keys():
        print '  ', clf_name
        # Get classifier by its name
        clf = classifiers[clf_name]['classifier']
        param_grid = classifiers[clf_name]['param_grid']
        
        # Find best params for classifier using CV
        cv_iterator = StratifiedKFold(y_train_res, n_folds=min(n_folds, sum(y_train_res)), shuffle=True)
        gscv = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, scoring=score,
            refit=False, cv=cv_iterator, n_jobs = 1, n_iter=n_iter)
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
    resample_multiplier_list = [1.25 + i * 0.25 for i in range(4*9)]
    n_folds_cv = 5 # to select good parameters of classifier using cross-validation
    n_folds_q = 20 # q - quality of classifier with resampling on dataset

    flag_short = False
    columns_data = ['dataset']
    columns_resample = ['resample_strategy', 'resample_multiplier', 'min_maj_ratio_final', 'q_iteration']
    columns_score = [i + '_' + j for i, j in itertools.product(classifiers.keys(), ScoreMetric._member_names_)]
    blank_columns = columns_data + columns_score + columns_resample
    results_blank = pd.DataFrame([np.zeros(len(blank_columns))], columns=blank_columns)
    results = pd.DataFrame(columns = blank_columns)
    
    filenames_good = [('data_csv/x%ddata.csv' % i) for i in [1,2,4,5] + range(8, 15) + [16, 18] + range(20, 26)]
    filenames_with_nans = [('data_csv/x%ddata.csv' % i) for i in [6, 15, 17, 19]]
    filenames_good_2 = [('data_csv/data2_%d.csv' % i) for i in range(96)]

    if len(sys.argv) == 1:
        filenames = filenames_good_2[52:57]
    else:
        if sys.argv[1] == '-short':
            flag_short = True
            filenames = ['data_csv/' + arg + '.csv' for arg in sys.argv[2:]]
        else:
            filenames = ['data_csv/' + arg + '.csv' for arg in sys.argv[1:]]
    
    if flag_short:
        filenames_str = filenames[0][9:-4] + '---' + filenames[-1][9:-4]
    else:
        filenames_str = '-'.join([s[9:-4] for s in filenames])
    #['data_csv/data_intrusion.csv']#[('data_csv/x%ddata.csv' % i) for i in [20, 21, 22, 23]]
    #[('data_csv/x%ddata.csv' % i) for i in [1]]#range(1, 31)]
    results.to_csv('exp_real_new_results_' + filenames_str + '.csv')

    for filename in filenames:
        print '  Work with file %s' % filename
        # Read data
        data = pd.read_csv(filename, index_col = 0)
        X = data[data.columns[:-1]]
        y = data['label']
        
        results_dataset = pd.DataFrame(columns=blank_columns)
        
        # Create train and test samples n_folds_q times
        q_iterator = StratifiedKFold(y, n_folds=min(n_folds_q, sum(y)), shuffle=True, random_state=RANDOM_STATE)
        q_iteration = 0
        for train_idx, test_idx in q_iterator:
            print 'train/test iteration %d' % q_iteration
            q_iteration += 1
            X_train = X.iloc[train_idx, :].values
            y_train = y.iloc[train_idx].values
            X_test = X.iloc[test_idx, :].values
            y_test = y.iloc[test_idx].values
        
            min_maj_rate_train = float(sum(y_train)) / (len(y_train) - sum(y_train))
            #print min_maj_rate_train

            # Standardize data
            scl = StandardScaler()
            scl.fit(X_train)
            X_train = scl.transform(X_train)
            X_test = scl.transform(X_test)

            # Do experiment for all resampling strategies, all resampling multipliers, all classifiers
            for resampling_name in ResamplingEnum._member_names_:
                print '\nResampling: %s' % resampling_name
                if resampling_name == 'nothing':
                    rml_temp = [1.0]
                else:
                    rml_temp = resample_multiplier_list

                for resample_multiplier in rml_temp:
                    print '%s, iteration %d, %s (multiplier=%.2f)' % (filename, q_iteration, resampling_name, resample_multiplier)
                    results_add = exp_once(X_train, y_train, X_test, y_test, resampling_name, resample_multiplier, results_blank, n_folds=n_folds_cv)
                    results_add['q_iteration'] = q_iteration
                    results_dataset = pd.concat([results_dataset, results_add], axis = 0)
        results_dataset['dataset'] = filename
        
        # Some sorting stuff
        results_dataset['sort_temp'] = results_dataset['resample_strategy'].replace({'nothing': 0, 'bootstrap': 1, 'rus': 2, 'smote': 3})
        results_dataset.sort(['sort_temp', 'resample_multiplier'], inplace = True)
        results_dataset.drop('sort_temp', 1, inplace = True)

        # Averaging results - not needed anymore, save results of all iterations.
        # results_dataset_av = pd.DataFrame(columns = blank_columns)
        # for resampling_name in ResamplingEnum._member_names_:
        #     df_temp1 = results_dataset.loc[results_dataset['resample_strategy'] == resampling_name]
        #     if resampling_name == 'nothing':
        #         rml_temp = [1.0]
        #     else:
        #         rml_temp = resample_multiplier_list
        #     for resample_multiplier in rml_temp:
        #         df_temp2 = df_temp1.loc[df_temp1['resample_multiplier'] == resample_multiplier]
        #         df_temp_add = df_temp2.iloc[[0],:].copy()
        #         df_temp_add[columns_score] = df_temp2[columns_score].mean(axis = 0).values
        #         results_dataset_av = pd.concat([results_dataset_av, df_temp_add], axis = 0)
        results = pd.concat((results, results_dataset), axis = 0)
        results.index = range(len(results))
        results.to_csv('exp_real_new_results_' + filenames_str + '.csv')
        
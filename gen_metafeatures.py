import sklearn
import pandas as pd
import numpy as np
import itertools
import scipy

from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from imbalanced import *
from imbalanced import _split_minor_major



def get_dist_class_centers(X, y):
    X_0 = X[y == 0]
    X_1 = X[y == 1]

    center_0 = X_0.mean().values
    center_1 = X_1.mean().values
    return np.linalg.norm(center_1 - center_0)

def get_mean_dist_between_classes(X, y, q_low = 0, q_high = 1):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    
    distances = []
    for a in X_0.values:
        for b in X_1.values:
            distances.append(np.linalg.norm(a-b))
    distances.sort()
    n_low = int(len(distances) * q_low)
    n_high = int(len(distances) * q_high)
    return np.mean(distances[n_low:n_high])


if __name__ == '__main__':

    results_blank = pd.DataFrame([['empty']], columns = ['dataset'])
    results = pd.DataFrame(data=None)
    df_all = pd.read_csv('./new_results/results_all.csv', index_col = 0)
    filenames_short = np.unique(df_all['dataset'].values).tolist()
    filenames = ['data_csv/' + f + '.csv' for f in filenames_short]

    for i in range(len(filenames)):
        filename = filenames[i]
        filename_short = filenames_short[i]
        print '  Work with file %s' % filename
        # Read data
        # Assume that labels are 0 and 1 - major and minor class respectively
        data = pd.read_csv(filename, index_col = 0, dtype = 'float64')
        data.index = range(len(data))
        X = data[data.columns.drop('label')]
        y = data['label']

        # Standardize data
        scl = StandardScaler()
        scl.fit(X)
        X = pd.DataFrame(scl.transform(X), columns = X.columns)

        results_dataset = results_blank.copy()

        results_dataset['dataset'] = filename_short
        # results_dataset['n_objects'] = X.shape[0]
        # results_dataset['n_features'] = X.shape[1]
        # results_dataset['objects_features_ratio'] = float(X.shape[0]) / X.shape[1]
        # results_dataset['min_maj_ratio'] = float(sum(y)) / (len(y) - sum(y))
        # results_dataset['dist_class_centers'] = get_dist_class_centers(X, y)
        # results_dataset['mean_dist_between_classes_0.0-0.3'] = get_mean_dist_between_classes(X, y, q_low = 0.0, q_high = 0.3)
        # results_dataset['mean_dist_between_classes_0.3-0.7'] = get_mean_dist_between_classes(X, y, q_low = 0.3, q_high = 0.7)
        # results_dataset['mean_dist_between_classes_0.7-1.0'] = get_mean_dist_between_classes(X, y, q_low = 0.7, q_high = 1.0)
        # results_dataset['max_major_cov_abs_eigenvalue'] = np.linalg.norm(np.cov(X.loc[y==0].T), ord = 2)
        # results_dataset['max_minor_cov_abs_eigenvalue'] = np.linalg.norm(np.cov(X.loc[y==1].T), ord = 2)
        # results_dataset['min_major_cov_abs_eigenvalue'] = np.linalg.norm(np.cov(X.loc[y==0].T), ord = -2)
        # results_dataset['min_minor_cov_abs_eigenvalue'] = np.linalg.norm(np.cov(X.loc[y==1].T), ord = -2)
        # results_dataset['major_cond_num'] = np.nan_to_num(np.linalg.cond(np.cov(X.loc[y==0].T)))
        # results_dataset['minor_cond_num'] = np.nan_to_num(np.linalg.cond(np.cov(X.loc[y==1].T)))
        # results_dataset['min_major_skewness'] = np.min(scipy.stats.skew(X.loc[y==0]))
        # results_dataset['min_minor_skewness'] = np.min(scipy.stats.skew(X.loc[y==1]))
        # results_dataset['max_major_skewness'] = np.max(scipy.stats.skew(X.loc[y==0]))
        # results_dataset['max_minor_skewness'] = np.max(scipy.stats.skew(X.loc[y==1]))
        results_dataset['min_major_skewness_stat'] = np.min(scipy.stats.skewtest(X.loc[y==0])[0])
        results_dataset['min_minor_skewness_stat'] = np.min(scipy.stats.skewtest(X.loc[y==1])[0])        
        results_dataset['max_major_skewness_stat'] = np.max(scipy.stats.skewtest(X.loc[y==0])[0])
        results_dataset['max_minor_skewness_stat'] = np.max(scipy.stats.skewtest(X.loc[y==1])[0])        
        # results_dataset['min_major_skewness_pvalue'] = np.min(scipy.stats.skewtest(X.loc[y==0])[1])
        # results_dataset['min_minor_skewness_pvalue'] = np.min(scipy.stats.skewtest(X.loc[y==1])[1])        
        # results_dataset['max_major_skewness_pvalue'] = np.max(scipy.stats.skewtest(X.loc[y==0])[1])
        # results_dataset['max_minor_skewness_pvalue'] = np.max(scipy.stats.skewtest(X.loc[y==1])[1])
        # results_dataset['min_major_kurtosis'] = np.min(scipy.stats.kurtosis(X.loc[y==0]))
        # results_dataset['min_minor_kurtosis'] = np.min(scipy.stats.kurtosis(X.loc[y==1]))
        # results_dataset['max_major_kurtosis'] = np.max(scipy.stats.kurtosis(X.loc[y==0]))
        # results_dataset['max_minor_kurtosis'] = np.max(scipy.stats.kurtosis(X.loc[y==1]))
        results_dataset['min_major_kurtosis_stat'] = np.min(scipy.stats.kurtosistest(X.loc[y==0])[0])
        results_dataset['min_minor_kurtosis_stat'] = np.min(scipy.stats.kurtosistest(X.loc[y==1])[0])
        results_dataset['max_major_kurtosis_stat'] = np.max(scipy.stats.kurtosistest(X.loc[y==0])[0])
        results_dataset['max_minor_kurtosis_stat'] = np.max(scipy.stats.kurtosistest(X.loc[y==1])[0])
        # results_dataset['min_major_kurtosis_pvalue'] = np.min(scipy.stats.kurtosistest(X.loc[y==0])[1])
        # results_dataset['min_minor_kurtosis_pvalue'] = np.min(scipy.stats.kurtosistest(X.loc[y==1])[1])
        # results_dataset['max_major_kurtosis_pvalue'] = np.max(scipy.stats.kurtosistest(X.loc[y==0])[1])
        # results_dataset['max_minor_kurtosis_pvalue'] = np.max(scipy.stats.kurtosistest(X.loc[y==1])[1])

        results = pd.concat((results, results_dataset), axis = 0)

    #scipy.stats.boxcox
    # for i, col in zip(range(results.shape[1]), results.columns):
    #     if col in ['dataset', 'n_objects', 'n_features', 'objects_features_ratio', 'min_maj_ratio']:
    #         continue
    #     col_min = results[col].min()
    #     col_max = results[col].max()
    #     if col_min < 0:
    #         continue
    #     if col_max < 100 * col_min:
    #         continue
    #     print col
    #     results[col] = np.log1p(results[col])
    #     results.columns = results.columns[:i].tolist() + [col + '_log1p'] + results.columns[i+1:].tolist()
        

    results.to_csv('metafeatures_new.csv')


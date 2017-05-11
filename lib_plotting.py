import numpy as np
import pandas as pd
import copy

#Helper functions for plotting
def get_dolanmore_points(df_scores, df_scores_all=None):
    # df_scores - scores of methods to be plotted
    # df_scores_all - scores of all methods; not used
    datasets = df_scores.index.tolist()
    methods = df_scores.columns.tolist()
    
    df_scores_best = df_scores.max(axis=1)
    df_scores_rel = 1 / df_scores.div(df_scores_best, axis=0) #how many times quality is worse than best_quality; >1
    beta_array = np.sort(np.unique(df_scores_rel.values))
    df_dm = pd.DataFrame(index = beta_array, columns = methods, data=0.0)
    
    for method in methods:
        scores_rel_counts = df_scores_rel[method].value_counts().sort_index()
        method_dm = pd.Series(index=beta_array, data=0.0)
        method_dm[scores_rel_counts.index] = scores_rel_counts.values
        method_dm = method_dm.cumsum() / len(datasets)
        df_dm[method] = method_dm
    return df_dm

def get_perfprofile_points(df_scores, df_scores_all=None):
    # df_scores - scores of methods to be plotted
    # df_scores_all - scores of all methods; not used
    datasets = df_scores.index.tolist()
    methods = df_scores.columns.tolist()
    
    q_array = np.sort(np.unique(np.append(df_scores.values.reshape(-1), [0.0, 1.0])))
    df_points = pd.DataFrame(index = q_array, columns = methods, data=0.0)
    
    for method in methods:
        scores_counts = df_scores[method].value_counts().sort_index()
        method_points = pd.Series(index=q_array, data=0.0)
        method_points[scores_counts.index] = scores_counts.values
        method_points = method_points.cumsum() / len(datasets)
        df_points[method] = method_points
    return df_points

def get_ara_points(df_scores, df_scores_all=None):
    # df_scores - scores of methods to be plotted
    # df_scores_all - scores of all methods, 
    # used to calculate max/min scores for each dataset
    datasets = df_scores.index.tolist()
    methods = df_scores.columns.tolist()
    
    scores_best = df_scores_all.max(axis=1)
    scores_worst = df_scores_all.min(axis=1)
    # ARA metric: (q_my - q_worst)/(q_best - q_worst) = 1 + (q_my - q_best)/(q_best - q_worst)
    df_scores_ara = 1 + df_scores.sub(scores_best, axis=0).div(
        (scores_best - scores_worst).replace(0,1), axis=0) 
    
    q_array = np.sort(np.unique(np.append(df_scores_ara.values.reshape(-1), [0.0, 1.0])))
    df_points = pd.DataFrame(index = q_array, columns = methods, data=0.0)
    for method in methods:
        scores_counts = df_scores_ara[method].value_counts().sort_index()
        method_points = pd.Series(index=q_array, data=0.0)
        method_points[scores_counts.index] = scores_counts.values
        method_points = method_points.cumsum() / len(datasets)
        df_points[method] = method_points  
    return df_points

# define options for graph plotting
line_styles = {'nothing': '-', 'recsyst1': '-', 'recsyst2': '-',
               'bootstrap': '-', 'rus': '-', 'smote1': '-', 'smote3': '-', 'smote5': '-', 'smote7': '-',
            'bootstrap_mcv': '-', 'rus_mcv': '-', 'smote1_mcv': '-', 'smote3_mcv': '-', 'smote5_mcv': '-', 'smote7_mcv': '-',
               'bootstrap_m=2.0': ':', 'rus_m=2.0': ':', 'smote1_m=2.0': ':', 'smote3_m=2.0': ':', 'smote5_m=2.0': ':', 
               'smote7_m=2.0': ':',
               'bootstrap_IR=1': '-', 'rus_IR=1': '-', 'smote1_IR=1': '-', 'smote3_IR=1': '-', 'smote5_IR=1': '-', 
               'smote7_IR=1': '-'}
line_dashes = {'nothing': [3,2,0.5,2], 'recsyst1': [], 'recsyst2': [],
               'bootstrap': [], 'rus': [], 'smote1': [], 'smote3': [], 'smote5': [], 'smote7': [],
               'bootstrap_mcv': [], 'rus_mcv': [], 'smote1_mcv': [], 'smote3_mcv': [], 'smote5_mcv': [], 'smote7_mcv': [],
               'bootstrap_m=2.0': [1,3], 'rus_m=2.0': [1,3], 'smote1_m=2.0': [1,3], 'smote3_m=2.0': [1,3], 'smote5_m=2.0': [1,3], 
               'smote7_m=2.0': [1,3],
               'bootstrap_IR=1': [1,1], 'rus_IR=1': [1,1], 'smote1_IR=1': [1,1], 'smote3_IR=1': [1,1], 'smote5_IR=1': [1,1], 
               'smote7_IR=1': [1,1]}
line_colors = {'nothing': 'grey', 'recsyst1': '#FF6600', 'recsyst2': '#DD00DD',
               'bootstrap': 'green', 'rus': 'red', 
               'smote1': '#0000F0', 'smote3': '#0000B0', 'smote5': '#000070', 'smote7': '#000030',
               'bootstrap_mcv': 'green', 'rus_mcv': 'red', 
               'smote1_mcv': '#0000F0', 'smote3_mcv': '#0000B0', 'smote5_mcv': '#000070', 'smote7_mcv': '#000030',
               'bootstrap_m=2.0': 'green', 'rus_m=2.0': 'red', 
               'smote1_m=2.0': '#0000F0', 'smote3_m=2.0': '#0000B0', 'smote5_m=2.0': '#000070', 'smote7_m=2.0': '#000030',
               'bootstrap_IR=1': 'green', 'rus_IR=1': 'red', 
               'smote1_IR=1': '#0000F0', 'smote3_IR=1': '#0000B0', 'smote5_IR=1': '#000070', 'smote7_IR=1': '#000030'}
clf_names = {'dtree': 'Decision tree', 'knn': r'$k$-NN', 'logreg_l1': r'$\ell_1$' + ' Log. regression'}
res_names = {'nothing': 'No resample', 'recsyst1': 'Rec. System 1', 'recsyst2': 'Rec. System 2',
             'bootstrap': 'ROS, CVS', 'rus': 'RUS, CVS', 
             'smote1': 'SMOTE, CVS', 'smote3': 'SMOTE, CVS', 'smote5': 'SMOTE, CVS', 'smote7': 'SMOTE, CVS',
             'bootstrap_mcv': 'ROS, m:CV', 'rus_mcv': 'RUS, m:CV', 
             'smote1_mcv': 'SMOTE, m:CV', 'smote3_mcv': 'SMOTE, m:CV', 'smote5_mcv': 'SMOTE, m:CV', 'smote7_mcv': 'SMOTE, m:CV',
           'bootstrap_m=2.0': 'ROS, m=2', 'rus_m=2.0': 'RUS, m=2',
           'smote1_m=2.0': 'SMOTE m=2', 'smote3_m=2.0': 'SMOTE m=2', 'smote5_m=2.0': 'SMOTE m=2', 'smote7_m=2.0': 'SMOTE m=2',
           'bootstrap_IR=1': 'ROS, EqS', 'rus_IR=1': 'RUS, EqS', 
           'smote1_IR=1': 'SMOTE EqS', 'smote3_IR=1': 'SMOTE EqS', 'smote5_IR=1': 'SMOTE, EqS', 'smote7_IR=1': 'SMOTE EqS'}
import numpy as np
import pandas as pd
import copy
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from lib_helper import *

class RecommendatorResBase():
    #X - df of metafeatures
    #Y - df of targets to fit on
    #Q - df of quality values
    #Q0 - df of quality values when default method is used
    def __init__(self, scale='standard'):
        self.scale_mode=scale
        
    def _scale_fit(self, X):
        if self.scale_mode == 'none':
            pass
        elif self.scale_mode == 'standard':
            self._scaler = StandardScaler()
            self._scaler.fit(X.values)
    
    def _scale_transform(self, X):
        if self.scale_mode == 'none':
            pass
        elif self.scale_mode == 'standard':
            X.loc[:,:] = self._scaler.transform(X.values)
    
    def _scale_fit_transform(self, X):
        self._scale_fit(X)
        self._scale_transform(X)
    
    def fit(self, X, Y, *args, **kwargs):
        self.resmult_pairs = Y.columns.tolist()
        if len(X) != len(Y):
            raise ValueError("Lengths of dataframes are not the same")
        if (X.index != Y.index).any():
            raise ValueError("indexes of dataframes are not the same")
        #print X.iloc[:5,:5]
        self._scale_fit_transform(X)
        #print X.iloc[:5,:5]
        #some fitting process  
    
    def recommend(self, X, *args, **kwargs):
        self._scale_transform(X)
        #some prediction procedure
    
    def assess_recommendation(self, rec, Q, Q0, *args, **kwargs):
        rec_scores = pd.DataFrame(data = 0.0, index=Q.index, 
            columns=['quality', 'gain_to_nores_abs', 'gain_to_nores_rel', 'gain_to_worst_abs', 'gain_to_best_abs', 
                     'ARA_metric'])
        quality_best = pd.concat((Q, Q0), axis=1).max(axis=1)
        quality_worst = pd.concat((Q, Q0), axis=1).min(axis=1)
        for ds in rec.index: #! possible error point: type of rec?
            if rec[ds] in Q.columns:
                rec_scores.loc[ds, 'quality'] = Q.loc[ds, rec[ds]]
            else:
                # Resampler is not recommended or recommended resampler is unknown 
                # => use default method (i.e. no-resampling)
                rec_scores.loc[ds, 'quality'] = Q0[ds]
                
        rec_scores['gain_to_nores_abs'] = rec_scores['quality'] - Q0
        rec_scores['gain_to_nores_rel'] = rec_scores['gain_to_nores_abs'].div(Q0)
        rec_scores['gain_to_worst_abs'] = rec_scores['quality'] - quality_worst
        rec_scores['gain_to_best_abs'] = rec_scores['quality'] - quality_best
        rec_scores['ARA_metric'] = 1 + (rec_scores['quality'] - quality_best) / (quality_best - quality_worst).replace(0, 1)
        
        return rec_scores
    
    def recommend_assess(self, X, Q, Q0, *args, **kwargs):
        #recommend and assess the recommendation
        rec = self.recommend(X, *args, **kwargs)
        rec_scores = self.assess_recommendation(rec, Q, Q0)
        return rec, rec_scores
    
    def fit_recommend_assess(self, X_train, Y_train, X_test, Q_test, Q0_test, *args, **kwargs):
        #fit recommendator, recommend methods and assess recommendation
        self.fit(X_train, Y_train, *args, **kwargs)
        return self.recommend_assess(X_test, Q_test, Q0_test, *args, **kwargs)
        
    def assess_cv(self, X, Y, Q, Q0, n_folds=10, random_state=1, *args, **kwargs):
        q_iterator = KFold(len(X), n_folds=n_folds, shuffle=True, random_state=random_state)
        rec_scores = pd.DataFrame(index=X.index, 
                              columns=['quality', 'gain_to_nores_abs', 'gain_to_nores_rel', 'gain_to_worst_abs', 
                                       'gain_to_best_abs', 'ARA_metric'])
        rec = pd.Series(index= X.index)
        for train_idx, test_idx in q_iterator:
            X_train, X_test = X.iloc[train_idx,:].copy(), X.iloc[test_idx,:].copy()
            Y_train = Y.iloc[train_idx,:].copy()
            Q_test = Q.iloc[test_idx,:].copy()
            Q0_test = Q0.iloc[test_idx]
        
            rec1, rec_scores1 = self.fit_recommend_assess(X_train, Y_train, X_test, Q_test, Q0_test, *args, **kwargs)
            rec_scores.loc[rec_scores1.index, rec_scores1.columns] = rec_scores1.values
            rec.loc[rec1.index] = rec1.values
        return rec, rec_scores

class RecommendatorRes1(RecommendatorResBase):

    def _get_labels(self, ser_scores): 
        if self.use_resampling_cond == 'less_than_t':
            return (ser_scores < self.threshold).astype(int)
        elif self.use_resampling_cond == 'more_than_t':
            return (ser_scores > self.threshold).astype(int)
        
    def __init__(self, scale='standard', threshold=0.05, use_resampling_cond='less_than_t'):
        RecommendatorResBase.__init__(self, scale)
        if use_resampling_cond in ['less_than_t', 'more_than_t']:
            self.use_resampling_cond = use_resampling_cond
        else:
            raise ValueError("Incorrect condition: '%s'. Possible conditions: less_than_t, more_than_t" % use_resampling_cond)
        self.threshold = threshold
        
    def fit(self, X, Y, base_clf=LogisticRegression(penalty='l2')):
        X, Y = X.copy(), Y.copy()
        RecommendatorResBase.fit(self, X, Y, base_clf=base_clf)
        #print X.iloc[:5,:5]
        
        self.classifiers = {}
        for resmult in Y.columns:
            y = self._get_labels(Y[resmult])
            clf = ModClassifier(base_clf)
            clf.fit(X, y)
            self.classifiers[resmult] = clf

    def predict_proba(self, X):
        return pd.DataFrame({resmult: self.classifiers[resmult].predict_proba(X)[:,1] for resmult in self.resmult_pairs}, 
                              index = X.index)
    
    def recommend(self, X, proba_threshold=0.5, **kwargs):
        # proba_threshold - minimum probability needed to use some non-default method
        X = X.copy()
        RecommendatorResBase.recommend(self, X, proba_threshold=proba_threshold)
        df_res_probas = self.predict_proba(X)
        df_res_probas['nothing'] = proba_threshold
        return df_res_probas.idxmax(axis = 1)
    
class RecommendatorRes2(RecommendatorResBase):
    # Classification-based RecommendatorRes1 for method recommendation
    # Regression models for multiplier recommendation
    def __init__(self, scale='standard', threshold=0.05, use_resampling_cond='less_than_t'):
        RecommendatorResBase.__init__(self, scale)
        self.threshold = threshold
        self.use_resampling_cond = use_resampling_cond
    
    def fit(self, X, Y, base_clf=LogisticRegression(penalty='l2'), base_rgr=DecisionTreeRegressor()):
        X, Y = X.copy(), Y.copy()
        RecommendatorResBase.fit(self, X, Y, base_clf=base_clf)
        self.res_names = Y.columns.levels[0].tolist()
        self.mult_values = Y.columns.levels[1].tolist()
        if self.use_resampling_cond == 'less_than_t':
            Y_with_best_mult = pd.DataFrame({res: Y[res].min(axis=1) for res in self.res_names})
            best_mults = {res: Y[res].idxmin(axis=1) for res in self.res_names}
        elif self.use_resampling_cond == 'more_than_t':
            Y_with_best_mult = pd.DataFrame({res: Y[res].max(axis=1) for res in self.res_names})
            best_mults = {res: Y[res].idxmax(axis=1) for res in self.res_names}          
        self.rec1_clf = RecommendatorRes1(scale=self.scale_mode, threshold=self.threshold, 
                                                use_resampling_cond=self.use_resampling_cond)
        self.rec1_clf.fit(X, Y_with_best_mult, base_clf=base_clf)
        self.rec2_regr_models = {res: copy.deepcopy(base_rgr) for res in self.res_names}
        for res in self.res_names:
            self.rec2_regr_models[res].fit(X, best_mults[res])
    
    def _recommend_mult(self, x, rec_method):
        if rec_method in self.res_names:
            #predicted = self.rec2_regr_models[rec_method].predict(X) #can be list or single value
            return self.rec2_regr_models[rec_method].predict(x.values.reshape(1,-1))[0]
        else: # resampling method is unknown or no resampling is recommended
            return 1.0
    def recommend(self, X, proba_threshold=0.5, **kwargs):
        rec_methods = self.rec1_clf.recommend(X, proba_threshold=proba_threshold, **kwargs)
        rec_mults = {ds: self._recommend_mult(X.loc[ds,:], rec_methods[ds]) for ds in X.index}
        return pd.Series({ds: (rec_methods[ds], rec_mults[ds]) for ds in X.index})
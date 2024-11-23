import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split, RandomizedSearchCV
#from sklearn import preprocessing
#from scipy.stats import randint
#from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from sklearn.base import clone
from sklearn.inspection import permutation_importance

SCORING = 'f1_macro'

RANDOM_STATE = 0

# Function to calculate permutation importance using sklearn
def calculate_permutation_importance(model, X_val, y_val, scoring=SCORING, n_repeats=5):
    result = permutation_importance(model, X_val, y_val, scoring=scoring,
                                    n_repeats=n_repeats, random_state=RANDOM_STATE)
    return result.importances_mean

# Custom RFE class with permutation importance
class RFEPermutationImportance:
    def __init__(self, estimator, n_features_to_select=None, step=1, min_features_to_select=1, n_repeats=5):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.n_repeats = n_repeats
        self.accuracies_ = []
        self.f1_scores_ = []

    def fit(self, X, y, X_test, y_test, features_lst):
        self.estimator_ = clone(self.estimator)
        
        features = list(features_lst)
        
        X_df = pd.DataFrame(X, columns=features)
        X_test_df = pd.DataFrame(X_test, columns=features)
        
        self.estimator_.fit(X_df[features], y)
        
        test_accuracy = accuracy_score(y_test, self.estimator_.predict(X_test_df[features]))
        self.accuracies_.append((len(features), test_accuracy))
        
        test_f1_score = f1_score(y_test, self.estimator_.predict(X_test[features]), average='macro')
        self.f1_scores_.append((len(features), test_f1_score))
        
        while len(features) > self.min_features_to_select:
            self.estimator_.fit(X[features], y)
            importance = calculate_permutation_importance(self.estimator_, X_df[features], y, n_repeats=self.n_repeats)
            
            # Identify least important feature
            least_important_feature_index = np.argmin(importance)
            least_important_feature = features[least_important_feature_index]
            
            # Remove the least important feature
            features.remove(least_important_feature)
            
            print(f'Removed feature: {least_important_feature}')
            print(f'Remaining features: {len(features)}')
            
            # Evaluate and store test accuracy and F1-score without removed feature
            
            self.estimator_.fit(X_df[features], y)
            
            test_accuracy = accuracy_score(y_test, self.estimator_.predict(X_test_df[features]))
            self.accuracies_.append((len(features), test_accuracy))
            
            test_f1_score = f1_score(y_test, self.estimator_.predict(X_test_df[features]), average='macro')
            self.f1_scores_.append((len(features), test_f1_score))
                    
        self.support_ = np.isin(X_df.columns, features)
        self.ranking_ = np.ones(len(X.columns), dtype=int)
        self.ranking_[~self.support_] = len(X_df.columns) - np.sum(self.support_) + 1
        
        return self

    def transform(self, X):
        return X.loc[:, self.support_]

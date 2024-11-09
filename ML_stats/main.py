import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

#from sklearn import preprocessing
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
#from collections import Counter
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV, KFold, StratifiedKFold

from features import init, init_blinks, init_blinks_quantiles
from features import init_blinks_no_head, init_blinks_no_head_quantiles

#columns_to_select = init
columns_to_select = init_blinks_no_head
#columns_to_select = init_blinks_no_head_quantiles
#columns_to_select = init_blinks
#columns_to_select = init_blinks_diam
#columns_to_select = init_blinks_quantiles

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

CHS = False
BINARY = False

#MODEL = "LR"
#MODEL = "SVC"
#MODEL = "RF"
MODEL = "HGBC"

N_ITER = 100
N_SPLIT = 10
SCORING = 'f1_macro'
#SCORING = 'accuracy'

#TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
TIME_INTERVAL_DURATION = 10
#TIME_INTERVAL_DURATION = 1

class ThresholdLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, percentiles=None):
        """
        Initialize with optional percentiles.
        If percentiles is None, perform no transformation.
        """
        self.percentiles = percentiles if percentiles else []
        self.thresholds = []

    def fit(self, X, y=None):
        # Calculate thresholds based on specified percentiles, if provided
        if self.percentiles:
            self.thresholds = [np.percentile(y, p * 100) for p in self.percentiles]
        return self
    
    def transform(self, X, y=None):
        if y is None:
            return X
        
        if not self.thresholds:
            # If no thresholds are specified, return y unchanged
            return X, y

        # Initialize all labels to the lowest class (1)
        y_transformed = np.ones(y.shape, dtype=int)
        
        # Apply thresholds to create labels
        for i, threshold in enumerate(self.thresholds):
            y_transformed[y >= threshold] = i + 2  # Increment class label for each threshold

        return X, y_transformed

# Function to perform parameter tuning with RandomizedSearchCV on each training fold
def model_with_tuning(pipeline, X_train, y_train):
    
    param_dist = get_param_dist()
    
    # Initialize RandomizedSearchCV with pipeline, parameter distribution, and inner CV
    randomized_search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=N_ITER, cv=N_SPLIT, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE
    )
    
    print("before randomized_search.fit")
    # Fit on the training data for this fold
    randomized_search.fit(X_train, y_train)
    print("after randomized_search.fit")
    
    # Return the best estimator found in this fold
    return randomized_search.best_estimator_


# Cross-validation function that handles the pipeline
def cross_val_with_label_transform(pipeline, X, y, cv):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for i, (train_index, test_index) in enumerate(cv.split(X), start=1):
        
        print(f"Iteration {i}")
        
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        
        pipeline.named_steps['label_transform'].fit(X_train, y_train)  # Fit to compute thresholds
        _, y_train_transformed = pipeline.named_steps['label_transform'].transform(X_train, y_train)
        _, y_test_transformed = pipeline.named_steps['label_transform'].transform(X_test, y_test)
        
        # Set class weights to the classifier
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')
        
        print("before model_with_tuning")
        # Get the best model after tuning on the current fold
        best_model = model_with_tuning(pipeline, X_train, y_train_transformed)
        print("after model_with_tuning")
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train_transformed)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        # Calculate the metrics
        accuracies.append(accuracy_score(y_test_transformed, y_pred))
        precisions.append(precision_score(y_test_transformed, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test_transformed, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_test_transformed, y_pred, average='macro', zero_division=0))
        
    # Print the results
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    

# Hold-out function that handles the pipeline
def hold_out_with_label_transform(pipeline, X, y):

    # Spit the data into train and test
    rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)
    
    for i, (train_idx, test_idx) in enumerate(rs.split(X)):
        X_train = np.array(X)[train_idx.astype(int)]
        y_train = np.array(y)[train_idx.astype(int)]
        X_test = np.array(X)[test_idx.astype(int)]
        y_test = np.array(y)[test_idx.astype(int)]
        
        pipeline.named_steps['label_transform'].fit(X_train, y_train)  # Fit to compute thresholds
        _, y_train_transformed = pipeline.named_steps['label_transform'].transform(X_train, y_train)
        _, y_test_transformed = pipeline.named_steps['label_transform'].transform(X_test, y_test)
        
        # Set class weights to the classifier
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')

        # Get the best model after tuning on the current fold
        best_model = model_with_tuning(pipeline, X_train, y_train_transformed)
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train_transformed)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        print("Shape at output after classification:", y_pred.shape)

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test_transformed)
        
        f1_macro = f1_score(y_pred=y_pred, y_true=y_test_transformed, average='macro')
        
        print("Accuracy:", accuracy)
        print("Macro F1-score:", f1_macro)
        

def get_model():
    if MODEL == "SVC":
        return SVC()
    elif MODEL == "RF":
        return RandomForestClassifier(random_state=RANDOM_STATE, max_features=None)
    else:
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)

def get_param_dist():
    if MODEL == "SVC":
        param_dist = {
            'classifier__C': uniform(loc=0, scale=10),  # Regularization parameter
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
            'classifier__gamma': ['scale', 'auto'],  # Kernel coefficient
            'classifier__degree': randint(1, 10)  # Degree of polynomial kernel
            }
    elif  MODEL == "RF":
       param_dist = {
            'classifier__n_estimators': randint(50,500),
            'classifier__max_depth': randint(1,79),
             #'min_samples_split': randint(2, 40),
             #'min_samples_leaf': randint(1, 40),
             #'max_features': ['auto', 'sqrt', 'log2', None],
             #'criterion': ['gini', 'entropy', 'log_loss']
            }
    else:
        param_dist = {
             'classifier__max_depth': randint(1,79),
             }
    return param_dist

def get_percentiles():
    if BINARY:
        return [0.93]
    else:
        return [0.52, 0.93]
    


def main():
    
    np.random.seed(RANDOM_STATE)
    print(f"RANDOM_STATE: {RANDOM_STATE}")
    
    if CHS:
        filename = "ML_features_CHS.csv"
    else:
        filename = "ML_features_" + str(TIME_INTERVAL_DURATION) + ".csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    data_df = data_df[['ATCO'] + columns_to_select]
    
    print(len(data_df.columns))
        
    if CHS:
        full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")
        scores_np = np.loadtxt(full_filename, delimiter=" ")
    else:
        full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

        scores_df = pd.read_csv(full_filename, sep=' ')
        scores_np = scores_df.to_numpy()
        
        #scores_np = np.loadtxt(full_filename, delimiter=" ")
    
        scores_np = scores_np[0,:] # Workload
    
    scores = list(scores_np)
    
    data_df['score'] = scores
    
    ###########################################################################
    
    data_df = data_df[data_df['ATCO']==3]
    data_df = data_df.drop('ATCO', axis=1)
    
    scores = data_df['score'].to_list()
    data_df = data_df.drop('score', axis=1)
    
    X = data_df.to_numpy()
    y = np.array(scores)
    
    zipped = list(zip(X, y))
    
    np.random.shuffle(zipped)
    
    X, y = zip(*zipped)
    
    X = np.array(X)
    y = np.array(y)
    
    pipeline = Pipeline([
            # Step 1: Standardize features
            ('scaler', StandardScaler()),
            # Step 2: Apply custom label transformation
            ('label_transform', ThresholdLabelTransformer(get_percentiles())),
            # Step 3: Choose the model
            ('classifier', get_model())
            ])
    
    
    # Initialize the cross-validation splitter
    #outer_cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    #cross_val_with_label_transform(pipeline, X, y, cv=outer_cv)
    
    hold_out_with_label_transform(pipeline, X, y)


start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import sys

#from sklearn import preprocessing
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
#from sklearn.utils.class_weight import compute_class_weight

from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from features import init_blinks_no_head
#from features import init, init_blinks
from features import left, right, average

#columns_to_select = init
#columns_to_select = init_blinks
columns_to_select = init_blinks_no_head

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
#ML_DIR = os.path.join(DATA_DIR, "MLInput_Journal")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0
CHS = False
BINARY = True

SUBJECT_SPECIFIC = False
ATCO = 0

RANDOM_SEARCH = True

LEFT_RIGHT_AVERAGE = False

#MODEL = "SVC"
#MODEL = "RF"
#MODEL = "BRF"
#MODEL = "EEC"
MODEL = "HGBC"

N_ITER = 100
N_SPLIT = 5
SCORING = 'f1_macro'

#TIME_INTERVAL_DURATION = 300
TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
#TIME_INTERVAL_DURATION = 10

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
        
        if CHS:
            if BINARY:
                y_transformed = [1 if score < 4 else 2 for score in y]
            else:
                y_transformed = [1 if score < 2 else 3 if score > 3 else 2 for score in y]
        else:
            # Apply thresholds to create labels
            for i, threshold in enumerate(self.thresholds):
                y_transformed[y >= threshold] = i + 2  # Increment class label for each threshold

        y_transformed = np.array(y_transformed)
        return X, y_transformed

# Function to perform parameter tuning with RandomizedSearchCV on each training fold
def model_with_tuning(pipeline, X_train, y_train):
    
    if not SUBJECT_SPECIFIC:
        #if MODEL != "EEC":
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')
    
    params = get_params()
    
    stratified_kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    
    if RANDOM_SEARCH:
        #search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=stratified_kfold, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=N_SPLIT, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        search = GridSearchCV(estimator=pipeline, param_grid=params, scoring=SCORING, cv=stratified_kfold, n_jobs=-1)
    
    # Fit on the training data for this fold
    search.fit(X_train, y_train)
    
    print("Best Parameters:", search.best_params_)
    
    # Return the best estimator found in this fold
    return search.best_estimator_

# Leave-one-out function that handles the pipeline
def leave_one_out_with_label_transform(pipeline, X, y):

    y_true_lst = []
    y_pred_lst = []
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)
    
    loo = LeaveOneOut()

    # Perform LOOCV
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_transformed[train_index], y_transformed[test_index]
        
        print(f"y_train: {y_train}, y_test: {y_test}")
        
        # Apply SMOTE to oversample the minority class
        #smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=1)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        
        print("before model_with_tuning")
        # Get the best model after tuning on the current fold
        best_model = model_with_tuning(pipeline, X_train, y_train)
        print("after model_with_tuning")
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        y_pred_lst.append(y_pred[0])
        y_true_lst.append(y_test[0])
    
    # Evaluate performance
    accuracy = accuracy_score(y_true_lst, y_pred_lst)
    balanced_acc = balanced_accuracy_score(y_true_lst, y_pred_lst)
    macro_precision = precision_score(y_true_lst, y_pred_lst, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_lst, y_pred_lst, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_lst, y_pred_lst, average='macro', zero_division=0)
    
    print(y_true_lst)
    print(y_pred_lst)
    
    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall: {macro_recall:.4f}")
    print(f"F1-Score: {macro_f1:.4f}")
    
    
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
    

# Stratified cross-validation function that handles the pipeline
def cross_val_stratified_with_label_transform(pipeline, X, y, cv):
    accuracies = []
    bal_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)

    
    for i, (train_index, test_index) in enumerate(cv.split(X, y_transformed), start=1):
        
        print(f"Iteration {i}")
        
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y_transformed)[train_index], np.array(y_transformed)[test_index]
        
        print(y_train)
        print(y_test)
        
        # Set class weights to the classifier
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')
        
        print("before model_with_tuning")
        # Get the best model after tuning on the current fold
        best_model = model_with_tuning(pipeline, X_train, y_train)
        print("after model_with_tuning")
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        print(y_test)
        print(y_pred)
        
        # Calculate the metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        bal_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
    # Print the results
    if SUBJECT_SPECIFIC:
        print(f"ATC0: {ATCO}")
    else:
        print("All ATCOs")
    
    print(f"Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
    print(f"Balanced accuracy: {np.mean(bal_accuracies):.2f} ± {np.std(bal_accuracies):.2f}")
    print(f"Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
    print(f"Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
    print(f"F1-Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")


# Hold-out function that handles the pipeline
def hold_out_with_label_transform(pipeline, X, y):
    
    # Spit the data into train and test
    ss = ShuffleSplit(n_splits=1, test_size=.1, random_state=RANDOM_STATE)
    
    for i, (train_idx, test_idx) in enumerate(ss.split(X,y)):
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
        
        #print("Shape at output after classification:", y_pred.shape)
        print(f"y_pred: {y_pred}")
        print(f"y_true: {y_test}")

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test_transformed)
        precision = precision_score(y_test_transformed, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_transformed, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_pred=y_pred, y_true=y_test_transformed, average='macro')
        
        print("Accuracy:", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Macro F1-score:", f1_macro)
        
# Hold-out function with stratified split that handles the pipeline
def hold_out_stratified_with_label_transform(pipeline, X, y):
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)


    # Spit the data into train and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=None)
    
    # Try several random states to ensure valid split
    for random_state in range(10):
        print(f"\nTrying random_state = {random_state}")
        try:
            # Perform the split
            for train_index, test_index in sss.split(X, y_transformed):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_transformed[train_index], y_transformed[test_index]
            
            # If the split was successful, print results
            print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
            print(f"Class distribution in train set: {Counter(y_train)}")
            print(f"Class distribution in test set: {Counter(y_test)}")
            break  # Exit the loop if a valid split is found
        
        except ValueError as e:
            print(f"Error: {e}")
    
    for i, (train_idx, test_idx) in enumerate(sss.split(X,y_transformed)):
        X_train = np.array(X)[train_idx.astype(int)]
        y_train = np.array(y_transformed)[train_idx.astype(int)]
        X_test = np.array(X)[test_idx.astype(int)]
        y_test = np.array(y_transformed)[test_idx.astype(int)]
        
        # Set class weights to the classifier
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')

        # Get the best model after tuning on the current fold
        best_model = model_with_tuning(pipeline, X_train, y_train)
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        #print("Shape at output after classification:", y_pred.shape)
        print(f"y_pred: {y_pred}")
        print(f"y_true: {y_test}")

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
        
        print("Accuracy:", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Macro F1-score:", f1_macro)

def get_model():
    print(f"Model: {MODEL}")
    if MODEL == "SVC":
        return SVC()
    elif MODEL == "RF":
        return RandomForestClassifier(random_state=RANDOM_STATE, max_features=None)
    elif MODEL == "BRF":
        return BalancedRandomForestClassifier(
                    bootstrap=False,
                    replacement=True,
                    sampling_strategy='auto',  # Under-sample majority class to match minority
                    random_state=RANDOM_STATE
                    )
    elif MODEL == "EEC":
        return EasyEnsembleClassifier(random_state=RANDOM_STATE)
    else:
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)

def get_params():
    if MODEL == "SVC":
        if RANDOM_SEARCH:
            params = {
                'classifier__C': uniform(loc=0, scale=10),  # Regularization parameter
                'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
                'classifier__gamma': ['scale', 'auto'],  # Kernel coefficient
                'classifier__degree': randint(1, 10)  # Degree of polynomial kernel
                }
        else:
            params_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 0.1],
                }
            
            params_set1 = {
                'classifier__C': [0.1],
                'classifier__kernel': ['linear'],
                }
            
            params_set2 = {
                'classifier__C': [0.01],
                'classifier__kernel': ['rbf'],
                'classifier__gamma': [0.1],
                }
            
            params_set3 = {
                'classifier__C': [0.001, 0.01, 1],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale'],
                }
            
            if SUBJECT_SPECIFIC:
                if ATCO == 1:
                    params = params_set1
                elif ATCO in [2, 4, 7, 8, 9, 10, 12, 13, 14, 18]:
                    params = params_set2
                elif ATCO in [5, 6, 11, 17]:
                    params = params_set3
                else:
                    print(f"ATCO{ATCO} is not supported")
                    sys.exit(0)
            else:
                params = params_grid

    elif  (MODEL == "RF") or (MODEL == "BRF"):
        if RANDOM_SEARCH:
            params = {
                'classifier__n_estimators': randint(50,500),
                'classifier__max_depth': randint(1,79),
                #'min_samples_split': randint(2, 40),
                #'min_samples_leaf': randint(1, 40),
                #'max_features': ['auto', 'sqrt', 'log2', None],
                #'criterion': ['gini', 'entropy', 'log_loss']
                }
        else:
            params_grid = {
                'classifier__n_estimators': [1, 5, 200, 500],
                'classifier__max_depth': [None, 1, 10, 79],
                }
            
            params_set1 = {
                'classifier__n_estimators': [200],
                'classifier__max_depth': [None, 1],
                }
            
            params_set2 = {
                'classifier__n_estimators': [1, 5,],
                'classifier__max_depth': [1, 10],
                }
            
            if SUBJECT_SPECIFIC:
                if ATCO in [1, 5, 6, 8, 9, 10, 12, 13, 18]:
                    params = params_set1
                elif ATCO in [2, 4, 6, 7, 11, 14, 17]:
                    params = params_set2
                else:
                    print(f"ATCO{ATCO} is not supported")
                    sys.exit(0)
            else:
                params = params_grid
    
    elif MODEL == "EEC":
       params = {
            'classifier__n_estimators': randint(50,500),
            }
    
    else: #HGBC
        
        if RANDOM_SEARCH:
            params = {
                'classifier__max_depth': randint(1,79),
                }
        else:
            shared_params = {
                # Maximum number of iterations (trees) to fit
                'classifier__max_iter': [100],
                # Maximum number of bins used for splitting
                'classifier__max_bins': [255],
                # Maximum depth of trees
                'classifier__max_depth': [None],
                # Maximum number of leaf nodes
                'classifier__max_leaf_nodes': [None],
                # L2 regularization parameter
                'classifier__l2_regularization': [0.0],
                # Number of iterations to wait before early stopping
                'classifier__n_iter_no_change': [1],
                # Scoring metric for early stopping
                'classifier__scoring': ['f1'],
                }
            
            params_grid = {
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__min_samples_leaf': [1, 20, 40],
                'classifier__early_stopping': ['auto', True],
                }
            combined_params_grid = {**shared_params, **params_grid}
            
            params_set1 = {
                'classifier__learning_rate': [0.1],
                'classifier__min_samples_leaf': [20],
                'classifier__early_stopping': ['auto'],
                }
            combined_params_set1 = {**shared_params, **params_set1}
            
            params_set2 = {
                'classifier__learning_rate': [0.01],
                'classifier__min_samples_leaf': [40],
                'classifier__early_stopping': ['auto'],
                }
            combined_params_set2 = {**shared_params, **params_set2}
            
            params_set3 = {
                'classifier__learning_rate': [0.01],
                'classifier__min_samples_leaf': [1],
                'classifier__early_stopping': [True],
                }
            combined_params_set3 = {**shared_params, **params_set3}
            
            if SUBJECT_SPECIFIC:
                if ATCO == 1:
                    params = combined_params_set1
                elif ATCO in [2, 4, 12, 14, 18]:
                    params = combined_params_set2
                elif ATCO in [5, 6, 7, 8, 9, 10, 11, 13, 17]:
                    params = combined_params_set3
                else:
                    print(f"ATCO{ATCO} is not supported")
                    sys.exit(0)
            else:
                params = combined_params_grid

    return params


def get_percentiles():
    if BINARY:
        print("BINARY")
        return [0.93]
    else:
        print("3 classes")
        return [0.52, 0.93]


def main():
    
    np.random.seed(RANDOM_STATE)
    print(f"RANDOM_STATE: {RANDOM_STATE}")
    print(f"Time interval: {TIME_INTERVAL_DURATION}")
    
    if CHS:
        filename = "ML_features_CHS.csv"
    else:
        filename = "ML_features_" + str(TIME_INTERVAL_DURATION) + ".csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    data_df = data_df[['ATCO'] + columns_to_select]
    
    if CHS:
        print("CHS")
        full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")
        scores_np = np.loadtxt(full_filename, delimiter=" ")
    else:
        print("EEG")
        full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

        scores_df = pd.read_csv(full_filename, sep=' ')
        scores_np = scores_df.to_numpy()
        
        #scores_np = np.loadtxt(full_filename, delimiter=" ")
    
        scores_np = scores_np[0,:] # Workload
    
    scores = list(scores_np)
    
    data_df['score'] = scores
    
    ###########################################################################
    
    print(f"Subject specific: {SUBJECT_SPECIFIC}")
    
    orig_num_slots = len(data_df.index)
    if SUBJECT_SPECIFIC:
        data_df = data_df[data_df['ATCO']==ATCO]
    
    print(f"Number of slots: {len(data_df.index)}")
    
    data_df = data_df.drop('ATCO', axis=1)
    
    if LEFT_RIGHT_AVERAGE:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[average[i]] = (data_df[col1] + data_df[col2])/2
            data_df = data_df.drop([col1, col2], axis=1)
    

    scores = data_df['score'].to_list()
    data_df = data_df.drop('score', axis=1)
    
    features = data_df.columns
    print(f"Number of features: {len(features)}")
    
    X = data_df.to_numpy()
    y = np.array(scores)
    
    zipped = list(zip(X, y))
    
    np.random.shuffle(zipped)
    
    X, y = zip(*zipped)
    
    X = np.array(X)
    y = np.array(y)
    
    if SUBJECT_SPECIFIC:
        pipeline = Pipeline([
                # Feature standartization step
                ('scaler', StandardScaler()),
                # Custom label transformation step
                ('label_transform', ThresholdLabelTransformer(get_percentiles())),
                # Oversampling step
                ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
                # Model setting step
                ('classifier', get_model())
                ])

        #leave_one_out_with_label_transform(pipeline, X, y)
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        cross_val_stratified_with_label_transform(pipeline, X, y, cv=outer_cv)
    else:
        pipeline = Pipeline([
                # Feature standartization step
                ('scaler', StandardScaler()),
                # Custom label transformation step
                ('label_transform', ThresholdLabelTransformer(get_percentiles())),
                # Model setting step
                ('classifier', get_model())
                ])

        # Initialize the cross-validation splitter
        #outer_cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        #cross_val_with_label_transform(pipeline, X, y, cv=outer_cv)
    
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        cross_val_stratified_with_label_transform(pipeline, X, y, cv=outer_cv)
    
        #hold_out_with_label_transform(pipeline, X, y)
        
        #hold_out_stratified_with_label_transform(pipeline, X, y)
    
        print(f"Number of slots: {len(data_df.index)}")
        print(f"Original number of slots: {orig_num_slots}")


start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    
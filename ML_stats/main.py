import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
import sys
import math

#from sklearn import preprocessing
from scipy.stats import randint, uniform
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score, jaccard_score, average_precision_score
from sklearn.metrics import confusion_matrix
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer
from sklearn.utils import resample
from sklearn.utils import shuffle
#from sklearn.utils.class_weight import compute_class_weight

from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.combine import  SMOTETomek
from imblearn.pipeline import Pipeline

from features import init_blinks_no_head
from features import init, init_blinks
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
ATCO = 17

RANDOM_SEARCH = True

LEFT_RIGHT_AVERAGE = False

MODEL = "LGBM"
#MODEL = "SVC"
#MODEL = "RF"
#MODEL = "BRF"
#MODEL = "EEC"
#MODEL = "HGBC"

BOOTSTRAP = False
N_ITER = 100
# Parameters for validation
if SUBJECT_SPECIFIC:
    N_SPLIT = 3 # Number of folds in stratified k-fold
    n_repeats = 5   # Number of times to repeat cross-validation
    n_bootstraps = 10  # Number of bootstrap samples per split
else:
    N_SPLIT = 10
SCORING = 'f1_macro'
#SCORING = make_scorer(average_precision_score)
#minority = make_scorer(recall_score, pos_label=2, zero_division=0)
#minority = make_scorer(precision_score, pos_label=2, zero_division=0)
#minority = make_scorer(f1_score, pos_label=2, zero_division=0)
#minority = make_scorer(jaccard_score, pos_label=2, zero_division=0)
#SCORING = minority


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

def calculate_classwise_accuracies(y_pred, y_true):
    
    print(f"y_pred: {y_pred}")
    print(f"y_true: {y_true}")
    
    y_pred_np = np.array(y_pred)
    y_true_np = np.array(y_true)
    
    # Get the unique classes
    if BINARY:
        classes = [1,2]
    else:
        classes = [1,2,3]

    # Find missing classes
    missing_classes = np.setdiff1d(classes, y_true_np)

    if missing_classes.size > 0:
        print("Missing classes:", missing_classes)
    else:
        print("All classes are present in y_true.")

    #print(classes)

    # Calculate accuracy for each class as a one-vs-all problem
    class_accuracies = []
    for cls in classes:
        # Get the indices of all instances where the class appears in y_true
        class_mask = (y_true_np == cls)
        
        class_accuracy = accuracy_score(y_true_np[class_mask], y_pred_np[class_mask])
        
        # Append the class accuracy
        class_accuracies.append(class_accuracy)

    #print(class_accuracies)
    
    return class_accuracies


# Function to perform parameter tuning with RandomizedSearchCV on each training fold
def model_with_tuning(pipeline, X_train, y_train):
    
    #if not SUBJECT_SPECIFIC:
    #if MODEL != "EEC":
    #    pipeline.named_steps['classifier'].set_params(class_weight='balanced')
    
    params = get_params()
    
    stratified_kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    
    if RANDOM_SEARCH:
        #search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=stratified_kfold, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=N_SPLIT, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        search = GridSearchCV(estimator=pipeline, param_grid=params, scoring=SCORING, cv=stratified_kfold, n_jobs=-1)
    
    # Fit on the training data for this fold
    search.fit(X_train, y_train)
    
    if SUBJECT_SPECIFIC:
        # Extract SMOTE-transformed training data
        smote = pipeline.named_steps['smote']
        _, y_resampled = smote.fit_resample(X_train, y_train)

        print("Resampled y_train:", y_resampled)
    
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
    for i, (train_index, test_index) in enumerate(loo.split(X), start=1):
      
        print(f"Iteration {i}")
    #for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_transformed[train_index], y_transformed[test_index]
        
        #if y_test!=2:
        #    continue
        
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
        
        # Check if the prediction matches the actual label
        if y_pred != y_test:
            print(f"Prediction mismatch for test sample {test_index}, redoing training...")
        
            X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
            # Fit the pipeline on transformed y_train
            best_model.fit(X_train, y_train)
            
            # Predict the labels on the transformed test data
            y_pred = best_model.predict(X_test)
        
        y_pred_lst.append(y_pred[0])
        y_true_lst.append(y_test[0])
        
    print(y_true_lst)
    print(y_pred_lst)
    
    # Evaluate performance
    accuracy = accuracy_score(y_true_lst, y_pred_lst)
    balanced_acc = balanced_accuracy_score(y_true_lst, y_pred_lst)
    c_w_accuracy = calculate_classwise_accuracies(y_pred=y_pred_lst, y_true=y_true_lst)
    macro_precision = precision_score(y_true_lst, y_pred_lst, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_lst, y_pred_lst, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_lst, y_pred_lst, average='macro', zero_division=0)
    
    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Class1 accuracy: {c_w_accuracy[0]:.4f}")
    print(f"Class2 accuracy: {c_w_accuracy[1]:.4f}")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall: {macro_recall:.4f}")
    print(f"F1-Score: {macro_f1:.4f}")
    
    y_true_np = np.array(y_true_lst)
    y_pred_np = np.array(y_pred_lst)
    print(y_true_np == y_pred_np)
    
    
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
    c_w_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)

    if BOOTSTRAP:
      for repeat in range(n_repeats):
        print(f"Repeat {repeat + 1}/{n_repeats}")
        for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y_transformed), start=1):
            print(f"  Fold {fold_idx + 1}/{N_SPLIT}")
        
            # Split into train and test sets
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y_transformed)[train_index], np.array(y_transformed)[test_index]
            
            # Bootstrap resampling within the training set
            for bootstrap in range(n_bootstraps):
                print(f"    Bootstrap {bootstrap + 1}/{n_bootstraps}")
            
                # Create a bootstrap sample
                X_train_resampled, y_train_resampled = resample(
                    X_train, y_train, replace=True, random_state=bootstrap)
                
                unique, counts = np.unique(y_train_resampled, return_counts=True)
                
                if (len(unique)>1) and (counts[1] >= 2):
            
                    # Fit the pipeline on the resampled training data
                    pipeline.fit(X_train_resampled, y_train_resampled)
                
                    # Set class weights to the classifier
                    pipeline.named_steps['classifier'].set_params(class_weight='balanced')
                
                    #print("before model_with_tuning")
                    # Get the best model after tuning on the current fold
                    best_model = model_with_tuning(pipeline, X_train, y_train)
                    #print("after model_with_tuning")
                
                    # Fit the pipeline on transformed y_train
                    best_model.fit(X_train, y_train)
                
                    # Predict the labels on the transformed test data
                    y_pred = best_model.predict(X_test)
                
                    print(y_test)
                    print(y_pred)
                
                    # Calculate the metrics
                    accuracies.append(accuracy_score(y_test, y_pred))
                    bal_accuracies.append(balanced_accuracy_score(y_test, y_pred))
                    c_w_accuracies.append(calculate_classwise_accuracies(y_pred=y_pred, y_true=y_test))
                    precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
                    recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
                    f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
                    
                else:
                    print("Skipping SMOTE due to insufficient minority samples.")
    else:
    
      for i, (train_index, test_index) in enumerate(cv.split(X, y_transformed), start=1):
        
        print(f"Iteration {i}")
        #if (i!=1) and (i!=9) and (i!=10):
        #    continue
        
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y_transformed)[train_index], np.array(y_transformed)[test_index]
        
        #print(y_train)
        #print(y_test)
        
        # Set class weights to the classifier
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')
        
        #print("before model_with_tuning")
        # Get the best model after tuning on the current fold
        best_model = model_with_tuning(pipeline, X_train, y_train)
        #print("after model_with_tuning")
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        print(y_test)
        print(y_pred)
        
        cm = confusion_matrix(y_test, y_pred, labels=[2, 1])
        print("Confusion Matrix:")
        print(cm)

        
        # Calculate the metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        bal_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        c_w_accuracies.append(calculate_classwise_accuracies(y_pred=y_pred, y_true=y_test))
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
    class1_values = [sublist[0] for sublist in c_w_accuracies if not math.isnan(sublist[0])]
    print(f"Class1 accuracy: {np.mean(class1_values):.2f} ± {np.std(class1_values):.2f}")
    class2_values = [sublist[1] for sublist in c_w_accuracies if not math.isnan(sublist[1])]
    print(f"Class2 accuracy: {np.mean(class2_values):.2f} ± {np.std(class2_values):.2f}")
    print(f"Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
    print(f"Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
    print(f"F1-Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
    print(f1_scores)


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
    if MODEL == "LGBM":
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    elif MODEL == "SVC":
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
    if MODEL == "LGBM":
        if RANDOM_SEARCH:
            params = {
                'classifier__num_leaves': np.arange(20, 150, 10),  # Range of number of leaves
                'classifier__max_depth': np.arange(3, 15),         # Range of max depth
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Learning rates to try
                'classifier__n_estimators': [100, 200, 500],      # Number of trees
                'classifier__min_data_in_leaf': [10, 20, 30],     # Minimum samples per leaf
                'classifier__lambda_l1': [0, 0.1, 1, 10],         # L1 regularization strength
                'classifier__lambda_l2': [0, 0.1, 1, 10],         # L2 regularization strength
                'classifier__feature_fraction': [0.7, 0.8, 0.9, 1.0],  # Feature sampling ratio
                'classifier__bagging_fraction': [0.7, 0.8, 0.9, 1.0],  # Row sampling ratio
                'classifier__bagging_freq': [0, 5, 10],           # Frequency of row sampling
                'classifier__subsample_for_bin': [200000, 300000, 400000],  # Number of samples used to construct histograms
                }
        else:
            params = {
                'classifier__num_leaves': [20, 31, 50],
                'classifier__max_depth': [5, 10, -1],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__n_estimators': [100, 200, 500],
                'classifier__min_data_in_leaf': [10, 20, 30],
                'classifier__lambda_l1': [0, 1],
                'classifier__lambda_l2': [0, 1],
                'classifier__feature_fraction': [0.8, 1.0],
                'classifier__bagging_fraction': [0.8, 1.0],
                'classifier__bagging_freq': [0, 5]
                }
    elif MODEL == "SVC":
        if RANDOM_SEARCH:
            params = {
                'classifier__C': uniform(loc=0, scale=1000),  # Regularization parameter
                #'classifier__kernel': ['linear', 'rbf', 'sigmoid'],  # Kernel type
                #'classifier__kernel': ['linear', 'rbf'],  # Kernel type
                'classifier__kernel': ['sigmoid', 'linear', 'rbf'],  # Kernel type
                #'classifier__kernel': ['rbf'],  # Kernel type
                #'classifier__gamma': ['scale', 'auto'],  # Kernel coefficient
                'classifier__gamma': ['auto', 0.001, 0.1],  # Kernel coefficient
                
                #'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                #'classifier__gamma': [1],
                #'classifier__coef0': uniform(-5, 5), # sigmoid
                #'classifier__tol': [0.0001],
                #'classifier__degree': randint(1, 10)  # Degree of polynomial kernel
                }
        else:
            '''
            test_params_grid = {
                #'classifier__C': [0.1, 1, 5, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                'classifier__C': [900],
                #'classifier__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'classifier__kernel': ['linear'],
                #'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__gamma': [100],
                #'classifier__coef0': [-5, -3, -1, 0, 1, 3, 5], # sigmoid
                #'classifier__degree': [1, 2, 3, 4, 5, 6]  # Degree of polynomial kernel
                }
            '''
            params_grid = {
                'classifier__C': [0.001, 0.01, 1, 5, 10, 100],
                #'classifier__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'scale', 'auto'],
                #'classifier__coef0': [-5, -3, -1, 0, 1, 3, 5], # sigmoid
                #'classifier__degree': [1, 2, 3, 4, 5, 6]  # Degree of polynomial kernel
                }
            
            params_set1 = {
                'classifier__C': [0.1],
                'classifier__kernel': ['linear'],
                }
            
            params_set2 = {
                'classifier__C': [5],
                #'classifier__degree': [6],
                'classifier__kernel': ['sigmoid'],
                'classifier__gamma': ['scale'],
                }
            
            params_set3 = {
                'classifier__C': [0.001, 0.01, 1],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale'],
                }
            
            params_set4 = {
                'classifier__C': [1],
                #'classifier__degree': [5],
                'classifier__kernel': ['sigmoid', 'rbf'],
                'classifier__gamma': ['auto'],
                }
            
            params_set5 = {
                'classifier__C': [5],
                #'classifier__degree': [5],
                'classifier__kernel': ['sigmoid', 'rbf'],
                'classifier__gamma': ['scale'],
                }
            
            params_set6 = {
                'classifier__C': [0.001],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': [0.001, 0.1],
                }
            
            params_set7 = {
                'classifier__C': [0.001, 0.01, 1, 10, 100],
                'classifier__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__coef0': [-5, -3, -1, 0, 1, 3, 5], # sigmoid
                'classifier__degree': [1, 2, 3, 4, 5, 6]  # Degree of polynomial kernel
                #'classifier__tol': [0.0001],
                }
            params_set8 = {
                'classifier__C': [ 548.8, 602.7],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['auto'],
                #'classifier__coef0': [-5, -3, -1, 0, 1, 3, 5], # sigmoid
                #'classifier__degree': [1, 2, 3, 4, 5, 6]  # Degree of polynomial kernel
                #'classifier__tol': [0.0001],
                }
            if SUBJECT_SPECIFIC:
                if ATCO in [5]: # only 5
                    params = params_set1
                elif ATCO in [4]:
                    params = params_set2
                elif ATCO in [10]:
                    params = params_set3
                elif ATCO in [7, 9, 11, 12]: # 1,2
                    params = params_set4
                elif ATCO in [6, 13, 14, 18]:
                    params = params_set5
                elif ATCO in [2, 8, 17]: # 1,2
                    params = params_set6
                elif ATCO in [1, 13]:
                    params = params_set7
                elif ATCO in [0]:
                    params = params_set8
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
        if RANDOM_SEARCH:
            params = {
                'classifier__n_estimators': randint(50,500),
            }
        else:
            params = {
                'classifier__n_estimators': [50, 100, 200, 400],
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
                if ATCO == 0: #1
                    params = combined_params_set1
                elif ATCO in [2, 4, 12, 14, 18]:
                    params = combined_params_set2
                elif ATCO in [1, 5, 6, 7, 8, 9, 10, 11, 13, 17]:
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

        scores_df = pd.read_csv(full_filename, sep=' ', header=None)
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
                #('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
                ('smote', BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
                #('smote', ADASYN(random_state=RANDOM_STATE, n_neighbors=1)),
                #('smote', SMOTETomek(smote=SMOTE(k_neighbors=1), random_state=RANDOM_STATE)),
                #('smote', SMOTEENN(smote=SMOTE(k_neighbors=1), random_state=RANDOM_STATE)),
                # Model setting step
                ('classifier', get_model())
                ])

        leave_one_out_with_label_transform(pipeline, X, y)
        #outer_cv = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
        #cross_val_stratified_with_label_transform(pipeline, X, y, cv=outer_cv)
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
        #outer_cv = KFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
        #cross_val_with_label_transform(pipeline, X, y, cv=outer_cv)
    
        outer_cv = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
        cross_val_stratified_with_label_transform(pipeline, X, y, cv=outer_cv)
    
        #hold_out_with_label_transform(pipeline, X, y)
        
        #hold_out_stratified_with_label_transform(pipeline, X, y)
    
        print(f"Number of slots: {len(data_df.index)}")
        print(f"Original number of slots: {orig_num_slots}")


start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    
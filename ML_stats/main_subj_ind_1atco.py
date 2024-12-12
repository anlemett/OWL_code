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

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans

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

RANDOM_SEARCH = True
#TEST_ATCO = 1

LEFT_RIGHT_AVERAGE = False

#MODEL = "LGBM"
MODEL = "SVC"
#MODEL = "RF"
#MODEL = "BRF"
#MODEL = "EEC"
#MODEL = "HGBC"

N_ITER = 100
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
        
    params = get_params()
    
    stratified_kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    
    if RANDOM_SEARCH:
        search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=stratified_kfold, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
        #search = RandomizedSearchCV(pipeline, params, n_iter=N_ITER, cv=N_SPLIT, scoring=SCORING, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        search = GridSearchCV(estimator=pipeline, param_grid=params, scoring=SCORING, cv=stratified_kfold, n_jobs=-1)
    
    # Fit on the training data for this fold
    search.fit(X_train, y_train)
        
    print("Best Parameters:", search.best_params_)
    
    # Return the best estimator found in this fold
    return search.best_estimator_
        
# Hold-out function with stratified split that handles the pipeline
def splitted_with_label_transform(pipeline, X_train, y_train, X_test, y_test):
    pipeline.named_steps['label_transform'].fit(X_train, y_train)  # Fit to compute thresholds
    _, y_train_transformed = pipeline.named_steps['label_transform'].transform(X_train, y_train)
    _, y_test_transformed = pipeline.named_steps['label_transform'].transform(X_test, y_test)
    
    # Create an undersampler, fitand resample the data
    #undersample = RandomUnderSampler(random_state=RANDOM_STATE)
    #X_train, y_train_transformed = undersample.fit_resample(X_train, y_train_transformed)
    
    #nm = NearMiss(version=1)
    #Version 1: Selects samples that are nearest neighbors to minority class samples.
    #Version 2: Selects samples that are farthest neighbors to minority class samples.
    #Version 3: Selects samples that are median neighbors to minority class samples.
    #X_train, y_train_transformed = nm.fit_resample(X_train, y_train_transformed)
    
    #s_s = 0.3
    #cc = ClusterCentroids(estimator=KMeans(n_clusters=2, random_state=RANDOM_STATE), sampling_strategy='auto', random_state=RANDOM_STATE)
    #cc = ClusterCentroids(sampling_strategy=s_s, random_state=RANDOM_STATE)
    #X_train, y_train_transformed = cc.fit_resample(X_train, y_train_transformed)

#def hold_out_stratified_with_label_transform(pipeline, X, y):
    
#    pipeline.named_steps['label_transform'].fit(X, y)  # Fit to compute thresholds
#    _, y_transformed = pipeline.named_steps['label_transform'].transform(X, y)


    # Spit the data into train and test
#    sss = StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=None)
    
    for i in range(1):
    #for i, (train_idx, test_idx) in enumerate(sss.split(X,y_transformed)):
        #X_train = np.array(X)[train_idx.astype(int)]
        #y_train = np.array(y_transformed)[train_idx.astype(int)]
        #X_test = np.array(X)[test_idx.astype(int)]
        #y_test = np.array(y_transformed)[test_idx.astype(int)]
        
        # Set class weights to the classifier
        pipeline.named_steps['classifier'].set_params(class_weight='balanced')
        #pipeline.named_steps['classifier'].set_params(class_weight={1: 1, 2: 10})

        # Get the best model after tuning on the current fold
        best_model = model_with_tuning(pipeline, X_train, y_train_transformed)
        
        # Fit the pipeline on transformed y_train
        best_model.fit(X_train, y_train_transformed)
        
        # Predict the labels on the transformed test data
        y_pred = best_model.predict(X_test)
        
        '''
        y_probs = best_model.predict_proba(X_test)
        threshold = 0.3
        y_pred = (y_probs[:, 1] >= threshold).astype(int)
        # Map predictions back to 1 and 2
        y_pred = np.where(y_pred == 1, 2, 1)
        
        # Get precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(y_test_transformed, y_probs[:, 1], pos_label=2)

        # Plot Precision-Recall Curve
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
        '''
        
        #print("Shape at output after classification:", y_pred.shape)
        print(f"y_pred: {y_pred}")
        print(f"y_true: {y_test_transformed}")

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test_transformed)
        bal_accuracy = balanced_accuracy_score(y_true=y_test_transformed, y_pred=y_pred)
        precision = precision_score(y_test_transformed, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_transformed, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_pred=y_pred, y_true=y_test_transformed, average='macro')
        
        print("Accuracy:", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Macro F1-score:", f1_macro)
        
        cm = confusion_matrix(y_test_transformed, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
    return (accuracy, bal_accuracy, precision, recall, f1_macro)

def get_model():
    print(f"Model: {MODEL}")
    if MODEL == "LGBM":
        return LGBMClassifier(random_state=RANDOM_STATE)
    elif MODEL == "SVC":
        #return SVC(probability=True)
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
                'classifier__C': uniform(loc=0, scale=10),  # Regularization parameter
                #'classifier__kernel': ['linear', 'rbf', 'sigmoid'],  # Kernel type
                #'classifier__kernel': ['linear', 'rbf'],  # Kernel type
                #'classifier__kernel': ['sigmoid', 'linear', 'rbf'],  # Kernel type
                'classifier__kernel': ['rbf'],  # Kernel type
                #'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
                'classifier__gamma': [0.0001, 0.001, 0.01, 0.1],  # Kernel coefficient
                #'classifier__gamma': ['auto', 0.001, 0.1],  # Kernel coefficient
                
                #'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                #'classifier__gamma': [1],
                #'classifier__coef0': uniform(-5, 5), # sigmoid
                #'classifier__tol': [0.0001],
                #'classifier__degree': randint(1, 5)  # Degree of polynomial kernel
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
            
            params = combined_params_grid

    return params


def get_percentiles():
    if BINARY:
        print("BINARY")
        return [0.93]
        #return [0.52]
    else:
        print("3 classes")
        return [0.52, 0.93]


def main(TEST_ATCO):
    
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
        #scores_np = scores_np[1,:] # Workload median
    
    scores = list(scores_np)
    
    data_df['score'] = scores
    
    ###########################################################################
        
    orig_num_slots = len(data_df.index)
    print(f"Number of slots: {len(data_df.index)}")
    
    #data_df = data_df.drop('ATCO', axis=1)
    
    if LEFT_RIGHT_AVERAGE:
        for i in range(0,17):
            
            col1 =  left[i]
            col2 = right[i]
            
            data_df[average[i]] = (data_df[col1] + data_df[col2])/2
            data_df = data_df.drop([col1, col2], axis=1)
    

    scores = data_df['score'].to_list()
    #data_df = data_df.drop('score', axis=1)
    
    data_df_train = data_df[data_df['ATCO']!=TEST_ATCO]
    data_df_test = data_df[data_df['ATCO']==TEST_ATCO]
    
    scores_train = data_df_train['score'].to_list()
    scores_test = data_df_test['score'].to_list()
       
    data_df_train = data_df_train.drop('ATCO', axis=1)
    data_df_train = data_df_train.drop('score', axis=1)
    data_df_test = data_df_test.drop('ATCO', axis=1)
    data_df_test = data_df_test.drop('score', axis=1)
    
    X_train = data_df_train.to_numpy()
    X_train = np.array(X_train)
    X_test = data_df_test.to_numpy()
    X_test = np.array(X_test)
    
    y_train = np.array(scores_train)
    y_test = np.array(scores_test)
    
    features = data_df.columns
    print(f"Number of features: {len(features)-2}") # -scores, ATCO
    
    X = data_df.to_numpy()
    y = np.array(scores)
    
    zipped = list(zip(X, y))
    
    np.random.shuffle(zipped)
    
    X, y = zip(*zipped)
    
    X = np.array(X)
    y = np.array(y)
    
    
    zipped = list(zip(X_train, y_train))

    np.random.shuffle(zipped)

    X_train, y_train = zip(*zipped)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    pipeline = Pipeline([
                # Feature standartization step
                ('scaler', StandardScaler()),
                # Custom label transformation step
                ('label_transform', ThresholdLabelTransformer(get_percentiles())),
                ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
                # Model setting step
                ('classifier', get_model())
                ])

    (accuracy, bal_accuracy, precision, recall, f1_macro) = splitted_with_label_transform(pipeline, X_train, y_train, X_test, y_test)
    
    print(f"Number of slots: {len(data_df.index)}")
    print(f"Original number of slots: {orig_num_slots}")
    return (accuracy, bal_accuracy, precision, recall, f1_macro)


start_time = time.time()

accuracies = []
bal_accuracies = []
precisions = []
recalls = []
f1_scores = []

#for TEST_ATCO in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]:
for TEST_ATCO in [2]:
    (accuracy, bal_accuracy, precision, recall, f1_macro) = main(TEST_ATCO)
    accuracies.append(accuracy)
    bal_accuracies.append(bal_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_macro)

print(f"Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Balanced accuracy: {np.mean(bal_accuracies):.2f} ± {np.std(bal_accuracies):.2f}")
print(f"Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
print(f"Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
print(f"F1-Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
print(f1_scores)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    
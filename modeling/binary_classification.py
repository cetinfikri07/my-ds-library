import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

# MODELLING
def classifiers():
    classifiers = {
        "LogisticRegression": LogisticRegression(random_state=0),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(random_state=0, probability=True),
        "CART": DecisionTreeClassifier(random_state=0),
        "AdaBoost": AdaBoostClassifier(random_state=0),
        "GBM": GradientBoostingClassifier(random_state=0),
        "RandomForest": RandomForestClassifier(random_state=0),
        "LGBM": LGBMClassifier(random_state=0),
        "CatBoost": CatBoostClassifier(random_state=0, verbose=False),
        "NaiveBayes": GaussianNB()
    }

    return classifiers

# MODEL EVAULATION
def strafied_kfolds(X_test,X,y,classifiers,kfolds):
    '''
    Applies stratifed K fold, prints out scores and predictions

            Parameters:
                    X_test (pd.Dataframe): test set
                    X (pd.Dataframe): train set
                    y (pd.Dataframe): train labels

    '''

    valid_scores = pd.DataFrame({"Classifier": classifiers.keys(),
                                 "Valid Score": np.zeros(len(classifiers)),
                                 "Training Time": np.zeros(len(classifiers))})
    i = 0
    for key,classifier in classifiers.items():
        cv = StratifiedKFold(n_splits=kfolds,shuffle=True,random_state=0)
        score = 0
        preds = np.zeros(len(X_test))
        start = time.time()
        for fold,(train_idx,val_idx) in enumerate(cv.split(X,y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
            clf = classifier
            # Train the model
            clf.fit(X,y)
            # make predictions and accuracy
            preds += clf.predict_proba(X_test)[:,1]
            score += clf.score(X_valid,y_valid)

        # Calculate average score
        score = score / kfolds
        valid_scores.iloc[i,1] = score

        # stop timer
        stop = time.time()
        valid_scores.iloc[i,2] = np.round((stop - start) / 60, 2)

        # Print accuracy and time
        print('Model:', key)
        print('Average validation accuracy:', np.round(100 * score, 2))
        print('Training time (mins):', np.round((stop - start) / 60, 2))
        print('')
        i += 1

    preds = preds / (kfolds * len(classifiers))

    return valid_scores,preds

# HYPERARAMETER TUNING
def classifiers_grid(grid_list):
    LR_grid = {'penalty': ['l1', 'l2'],
               'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
               'max_iter': [50, 100, 150]}

    KNN_grid = {'n_neighbors': [3, 5, 7, 9],
                'p': [1, 2]}

    SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']}

    CART_grid = {"criterion" : ['gini', 'entropy'],"max_depth" : [2,4,6,8,10,12]}

    AdaBoost_grid = {"learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                       "n_estimators": [10, 50, 100, 500]}

    GBM_grid = {"learning_rate": [0.25, 0.1, 0.05, 0.01],
                  "n_estimators": [1, 2, 4, 8, 16, 32, 64, 100, 200],
                  "max_depth": [3, 8, 10, 12, 14],
                  "min_samples_split": [2, 3, 4],
                  }

    RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
               'max_depth': [4, 6, 8, 10, 12]}

    LGBM_grid = {"learning_rate": [0.25, 0.1, 0.05, 0.01],
                       "max_depth": [-1, 1, 2, 3, 4, 5],
                       "num_leaves": [10, 20, 30, 40, 50],
                       "n_estimators": [100, 250, 300, 350, 500, 1000],
                       "colsample_bytree": [0.5, 0.8, 0.7, 0.6, 1]}

    CatBoost_grid = {"iterations": [200, 300, 400],
                       "learning_rate": [0.01, 0.1],
                       "depth": [3, 6]}

    NB_grid = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}

    grid = {
        "LogisticRegression": LR_grid,
        "KNN": KNN_grid,
        "SVC": SVC_grid,
        "CART" : CART_grid,
        "AdaBoost" : AdaBoost_grid,
        "GBM" : GBM_grid,
        "RandomForest": RF_grid,
        "LGBM": LGBM_grid,
        "CatBoost": CatBoost_grid,
        "NaiveBayes": NB_grid
    }

    return {x: grid[x] for x in grid_list}

# GRIDSEARCH
def gridSearchCV(X,y,grid_list):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    clf_best_params = {x: classifiers[x] for x in grid_list}

    valid_scores = pd.DataFrame({"Classifier": clf_best_params.keys(),
                                 "Valid Score": np.zeros(len(clf_best_params)),
                                 "Training Time": np.zeros(len(clf_best_params))})
    i = 0
    for key, classifier in clf_best_params.items():
        start = time.time()
        clf = GridSearchCV(estimator=classifier, param_grid=grid[key], n_jobs=-1)

        # Train and score
        clf.fit(X_train, y_train.values.ravel())
        valid_scores[i, 1] = clf.score(X_valid, y_valid.values.ravel())

        # save trained model with best params
        clf_best_params[key] = clf.best_params_

        # Print iteration and training time
        stop = time.time()
        valid_scores[i, 2] = np.round((start - stop) / 60, 2)

        print('Model:', key)
        print('Training time (mins):', valid_scores.iloc[i, 2])
        print('')
        i += 1

    return valid_scores,clf_best_params

# PREDICTIONS
def plot_predictions(preds):
    # Plot predictions
    plt.figure(figsize=(10,4))
    sns.histplot(preds,binwidth=0.01,kde=True)
    plt.title('Predicted probabilities')
    plt.xlabel('Probability')


# METRICS
def plot_confusion_matrix(y_true,y_pred,percentage = False):
    acc = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)

    text = f"""Accuracy score : {acc:.2f}
    Precision score: {precision:.2f}
    Recall score : {recall:.2f}
    f1 score: {f1:.2f}"""

    # Figure
    fig = plt.figure(figsize = (10,8))
    cm = confusion_matrix(y_true,y_pred) / np.sum(confusion_matrix(y_true,y_pred)) if percentage else confusion_matrix(y_true,y_pred)
    fmt = ".2%" if percentage else "g"
    sns.heatmap(cm,annot=True,fmt=fmt, cmap='Blues')
    plt.xlabel("Predicted labels")
    plt.ylabel("True label")
    plt.figtext(0.5,0.01,text,ha="center",fontsize = 18,bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.show()

# POST PROCESSING
# Proportion of predicted positive classes
def preds_prop(preds_arr,thresh):
    pred_classes = (preds_arr >= thresh).astype(int)
    return pred_classes.sum()/len(pred_classes)

# Plot proportions across a range of thresholds and find optimal threshold
def plot_preds_prop(preds_arr):
    # Array of thresholds
    T_array = np.arange(0, 1, 0.001)

    # Calculate proportions
    prop = np.zeros(len(T_array))
    for i, T in enumerate(T_array):
        prop[i] = preds_prop(preds_arr, T)

    # Plot proportions
    plt.figure(figsize=(10, 4))
    plt.plot(T_array, prop)
    target_prop = 0.519  # Experiment with this value
    plt.axhline(y=target_prop, color='r', linestyle='--')
    plt.text(-0.02, 0.45, f'Target proportion: {target_prop}', fontsize=14)
    plt.title('Predicted target distribution vs threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Proportion')

    # Find optimal threshold (the one that leads to the proportion being closest to target_prop)
    T_opt = T_array[np.abs(prop - target_prop).argmin()]
    print('Optimal threshold:', T_opt)
    return T_opt

# Classify test set using optimal threshold
def predict_with_optimal_threshold(X,preds,T_opt,col_name):
    preds_tuned = (preds>=T_opt).astype(int)
    X[col_name] = preds_tuned
    return X



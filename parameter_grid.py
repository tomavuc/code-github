import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost
import sklearn


models_dict = {
    # "XGB": [
    #     XGBClassifier(device = "cuda", random_state=42, verbosity = 0),
    #     {"n_estimators": [100, 150, 200],
    #     "max_depth": [None, 2, 5, 10],
    #     "learning_rate": [0.001, 0.01, 0.1, 1],
    #     "min_child_weight": [1, 5, 10],
    #     "eval_metric": ["auc", "aucpr", "logloss"], 
    #     "alpha": [0, 0.2, 0.5, 1]}
    # ], 
    "KNN": [
        KNeighborsClassifier(),
        {"n_neighbors": [1, 2, 3, 5],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]}
    ],
    "SGD": [
        SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        {"alpha": [0.0001, 0.001, 0.01],
        "loss": ["hinge", "log_loss", "huber"],
        "penalty": ['l2', 'l1', 'elasticnet'],
        "learning_rate": ['constant', 'optimal', 'invscaling'],
        "eta0": [0.00001, 0.01, 0.1, 1.0],
        "early_stopping": [True, False]}
    ],
    "SVM": [
        SVC(random_state=42),
        {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10],
        "kernel": ["rbf", "linear", "poly"],
        "degree": [3, 5, 7],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"]}
    ],
    "DT": [
        DecisionTreeClassifier(random_state=42),
        {"criterion": ["gini", "log_loss", "entropy"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 3, 5],
        "max_features": ["sqrt", "log2", None, 30, 50, 70, 90, 10, 20]}
    ],
    "RF": [
       RandomForestClassifier(random_state=42),
       {"criterion": ["gini", "log_loss", "entropy"],
       "n_estimators": [100, 150, 200],
       "max_depth": [None, 5, 10, 15],
       "min_samples_split": [2, 3, 5],
       "max_features": ["sqrt", "log2", None, 30, 50, 70, 90, 10, 20]}
    ]
    #"XGB": [
        #XGBClassifier(device = "cuda", gpu_id = 0, random_state=42),
        #{"classifier__n_estimators": [100, 150, 200],
        #"classifier__max_depth": [3, 4, 5],
        #"classifier__learning_rate": [0.01, 0.1, 0.2, 0.4],
        #"classifier__min_child_weight": [1, 5, 10],
        #"classifier__eval_metric": ["auc", "aucpr", "logloss"]}]
}

d = xgboost.DMatrix(np.random.randn(100,10), label=np.random.randint(0,2,100))
xgboost.train({"device": "cuda"}, d, num_boost_round=1)
print("scikit-learn version:", sklearn.__version__)
print("loaded from         :", sklearn.__file__)
print(xgboost.__version__)
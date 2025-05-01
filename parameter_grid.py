import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC

models_dict = {
    "KNN": [
        KNeighborsClassifier(),
        {"classifier__n_neighbors": [1, 2, 3, 5],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"]},
    ],
    "SGD": [
        SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        {"classifier__alpha": [0.0001, 0.001, 0.01],
        "classifier__loss": ["hinge", "log_loss", "huber"],
        "classifier__penalty": ['l2', 'l1', 'elasticnet'],
        "classifier__learning_rate": ['constant', 'optimal', 'invscaling'],
        "classifier__eta0": [0, 0.01, 0.1, 1],
        "classifier__early_stopping": [True, False]},
    ],
    "SVM": [
        SVC(random_state=42),
        {"classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf", "linear", "poly"],
        "classifier__degree": [3, 5, 7],
        "classifier__gamma": ["scale", "auto"],},
    ],
    "DT": [
        DecisionTreeClassifier(random_state=42),
        {"classifier__criterion": ["gini", "log_loss", "entropy"],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 3, 5],
        "classifier__max_features": ["sqrt", "log2", None]
        },
    ],
    "RF": [
        RandomForestClassifier(random_state=42),
        {"classifier__criterion": ["gini", "log_loss", "entropy"],
        "classifier__n_estimators": [50, 100, 150, 200],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 3, 5],
        "classifier__max_features": ["sqrt", "log2", None]},
    ]#,
    #"XGB": [
        #XGBClassifier(eval_metric='logloss', random_state=42),
        #{"classifier__n_estimators": [1, 2, 5],
        # "classifier__max_depth": [3, 5, 7],
        #"classifier__learning_rate": [0.01, 0.1, 0.2],},],
}

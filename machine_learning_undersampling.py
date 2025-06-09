import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report, accuracy_score

from parameter_grid import models_dict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

from utils import balance_and_split_data

print("Machine learning with undersampling technique, tested on only negative data")
balanced, remaining = balance_and_split_data('all_merged_v2.csv')

missing_per_column = balanced.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill in the balanced matrix!')
else:
    print('Data is full!')

X_train = balanced.drop(columns=['id','UniProtID','pIspG','organism']) 
y_train = balanced['pIspG'].replace({'+/-': '+'})
y_train = y_train.replace({'+': 1, '-': 0})
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
var_thresh = 0.1
X_train_filtered = X_scaled

print(X_train_filtered)
print(y_train)

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 1/3, random_state=40)

X_test = remaining.drop(columns=['id','UniProtID','pIspG','organism']) 
y_test = remaining['pIspG']
y_test = y_test.replace({'-': 0})
missing_per_column = remaining.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill!')
else:
    print('Data is full!')
print(X_test)
print(y_test)
scaler2 = StandardScaler()
X_scaled2 = pd.DataFrame(scaler2.fit_transform(X_test), columns=X_test.columns, index=X_test.index)
X_test_filtered = X_scaled2

print(X_test_filtered)

def tune_and_evaluate(model, param_grid, pipeline_name):
    
    grid = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_filtered, y_train)
    
    print(f"\nBest parameters for {pipeline_name}: {grid.best_params_}")
    best_model = grid.best_estimator_
    
    train_preds = best_model.predict(X_train_filtered)
    test_preds = best_model.predict(X_test_filtered)
    
    print(f"{pipeline_name} - Train Accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"{pipeline_name} - Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print("Classification Report for Test Data:")
    print(classification_report(y_test, test_preds))
    
    return best_model

param_grid_knn = {
    'n_neighbors': [1, 3, 5],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

param_grid_sgd = {
    'loss': ['hinge', 'log_loss'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'penalty': ['l2', 'l1', 'elasticnet']
}

param_grid_rf = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 2, 5],
    'max_features': ['sqrt', 'log2']
}

param_grid_dt = {
    'max_depth': [None, 2, 5],
    'min_samples_split': [2, 5, 10]
}

param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

param_grid_xgboost = {"n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.1, 0.2, 0.4],
        "min_child_weight": [1, 5, 10],
        "eval_metric": ["auc", "aucpr", "logloss"]}

print("Tuning KNN:")
best_knn = tune_and_evaluate(KNeighborsClassifier(), param_grid_knn, "KNN")

print("Tuning SGD:")
best_sgd = tune_and_evaluate(SGDClassifier(random_state=42), param_grid_sgd, "SGD")

print("Tuning Random Forest:")
best_rf = tune_and_evaluate(RandomForestClassifier(random_state=42), param_grid_rf, "RF")

print("Tuning Decision Tree:")
best_dt = tune_and_evaluate(DecisionTreeClassifier(random_state=42), param_grid_dt, "DT")

print("Tuning SVC:")
best_svc = tune_and_evaluate(SVC(random_state=42), param_grid_svc, "SVC")

print("Tuning XGBoost:")
best_xgb = tune_and_evaluate(XGBClassifier(device = "cuda", random_state=42), param_grid_xgboost, "XGB")

best_models = {"XGB": best_xgb, "KNN": best_knn, "SGD": best_sgd, "RF": best_rf, "DT": best_dt, "SVM": best_svc}

joblib.dump(best_models, "best_models_undersampling.pkl")
print("Saved one best estimator per model class to artifacts/best_models.pkl")
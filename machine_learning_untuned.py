import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

data = pd.read_csv('all_merged.csv')
data['pIspG'] = data['pIspG'].replace({'+/-': '+'})
data['pIspG'] = data['pIspG'].replace({'+': 1, '-': 0})

missing_per_column = data.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill!')
else:
    print('Data is full!')

X = data.drop(columns=['id','UniProtID','pIspG','organism'])
y = data['pIspG']

pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

base_models = [XGBClassifier(device = "cuda", random_state=42, scale_pos_weight=pos_weight), 
               KNeighborsClassifier(),
               SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
               SVC(random_state=42),
               DecisionTreeClassifier(random_state=42),
               RandomForestClassifier(random_state=42)]
    
for model in base_models:    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)])

    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1)

    print(f"Balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    print("\nClassification report:\n", classification_report(y, y_pred, zero_division=0))

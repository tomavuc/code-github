import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report

from parameter_grid import models_dict

#Data Loading (After generation)
data = pd.read_csv('all_merged.csv')
data.head()
data['pIspG'] = data['pIspG'].replace({'+/-': '+'})
data['pIspG'] = data['pIspG'].replace({'+': 1, '-': 0})

missing_per_column = data.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill!')
else:
    print('Data is full!')

X = data.drop(columns=['id','UniProtID','pIspG','organism'])
top_feats = ['length', 'molecular_weight', 'pid_total', 'irc_negative_positive_sum', 'aa_S', 'res_type_7_pct', 'isoelectric_point', 'aa_L', 'res_charge_sum', 'CL_FLDA_B', 'CL_FLDA_A', 'GO:0046872', 'GO:0005506', 'RMSE', 'aa_Q', 'polarity_2_pct', 'penalty_top50', 'res_type_9_pct', 'aa_R', 'res_charge_negative_sum']
X = X[top_feats] 
y = data['pIspG']
#y = LabelEncoder().fit_transform(y)
print(y)

outer_cv = KFold(n_splits = 5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits = 4, shuffle=True, random_state=42)
results = {}
grid_objects = {}

#Full pipeline (first try)
for name, (model, params) in models_dict.items():
    pipe = Pipeline([('scaler', StandardScaler()), ("classifier", model)])

    print(f"Grid search started for {name}")
    grid = GridSearchCV(pipe, param_grid = params, cv = inner_cv, scoring = "balanced_accuracy", n_jobs = -1)

    scores = cross_val_score(grid, X, y, cv = outer_cv, scoring = "balanced_accuracy", n_jobs = -1)

    results[name] = {"mean_accuracy": np.mean(scores), "std_accuracy": np.std(scores)}
    grid_objects[name] = grid 
    print(f"Model: {name}, Mean Accuracy: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")

    y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

best_name = max(results, key=lambda k: results[k]["mean_accuracy"])
print(f"Best‑performing model: {best_name} - ({results[best_name]['mean_accuracy']:.4f} ± {results[best_name]['std_accuracy']:.4f})")
best_grid = grid_objects[best_name]
best_grid.fit(X, y)

best_params = best_grid.best_params_
print(f"\nBest hyper‑parameters: {best_params}")

print("\nNested CV Results:")
for model_name, metrics in results.items():
    print(model_name, metrics)


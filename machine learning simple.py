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

data = pd.read_csv('datasets/features.csv')
final_feature_matrix = pd.read_csv('datasets/final_feature_matrix.csv')

existing_ids = set(data['GenBankID'])

new_data = final_feature_matrix[~final_feature_matrix['GenBankID'].isin(existing_ids)]
new_data.to_csv('datasets/new_data_only.csv', index=False)


missing_per_column = data.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill!')
else:
    print('Data is full!')


X_train = data.drop(columns=['GenBankID','UniProtID','pIspG','organism']) 
y_train = data['pIspG']
print(X_train)
print(y_train)

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 1/3, random_state=40)

data2 = pd.read_csv('datasets/new_data_only.csv')
X_test = data2.drop(columns=['GenBankID','UniProtID','pIspG','organism']) 
y_test = data2['pIspG']
missing_per_column = data2.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill!')
else:
    print('Data is full!')
print(X_test)
print(y_test)

pipe = Pipeline([('scaler', StandardScaler())])

X_train_scaled = pipe.fit_transform(X_train)
X_test_scaled = pipe.transform(X_test)

# def train_and_evaluate(model, X_train, X_test, y_train, y_test, pipeline_name):
#     model.fit(X_train, y_train)
#     train_predictions = model.predict(X_train)
#     test_predictions = model.predict(X_test)

#     train_acc = accuracy_score(y_train, train_predictions)
#     test_acc = accuracy_score(y_test, test_predictions)

#     print(f'\n{pipeline_name} - Train MAE: {train_acc:.4f}, Test MAE: {test_acc:.4f}')
#     print("\nClassification Report for Test Data:")
#     print(classification_report(y_test, test_predictions))
#     return model

def tune_and_evaluate(model, param_grid, pipeline_name):
    # Create a pipeline that includes scaling and the classifier
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy = 'mean')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"\nBest parameters for {pipeline_name}: {grid.best_params_}")
    best_model = grid.best_estimator_
    
    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)
    
    print(f"{pipeline_name} - Train Accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"{pipeline_name} - Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print("Classification Report for Test Data:")
    print(classification_report(y_test, test_preds))
    
    return best_model

param_grid_knn = {
    'classifier__n_neighbors': [1, 3, 5],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

param_grid_sgd = {
    'classifier__loss': ['hinge', 'log_loss'],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__penalty': ['l2', 'l1', 'elasticnet']
}

param_grid_rf = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__max_depth': [None, 2, 5],
    'classifier__max_features': ['sqrt', 'log2']
}

param_grid_dt = {
    'classifier__max_depth': [None, 2, 5],
    'classifier__min_samples_split': [2, 5, 10]
}

param_grid_svc = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

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



# # KNN Regressor
# print('For KNN:')
# train_and_evaluate(KNeighborsClassifier(), X_train_scaled, X_test_scaled, y_train, y_test, 'Pipeline')

# # SGD Regressor
# print('\nFor SGD:')
# train_and_evaluate(SGDClassifier(), X_train_scaled, X_test_scaled, y_train, y_test, 'Pipeline')

# # Random Forest Regressor
# print('\nFor Random Forest:')
# train_and_evaluate(RandomForestClassifier(), X_train_scaled, X_test_scaled, y_train, y_test, 'Pipeline')

# #Decision Tree Regressor
# print('\nFor decision tree:')
# train_and_evaluate(DecisionTreeClassifier(), X_train_scaled, X_test_scaled, y_train, y_test, 'Pipeline')

# #Dummy regressor
# print('\nThe dummy classifier would predict: ')
# train_and_evaluate(SVC(), X_train_scaled, X_test_scaled, y_train, y_test, 'Dummy 1')
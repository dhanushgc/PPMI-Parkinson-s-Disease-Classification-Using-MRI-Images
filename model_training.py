import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

X_train = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\X_train_features.npy")
y_train = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\y_train.npy")
X_valid = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\X_valid_features.npy")
y_valid = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\y_valid.npy")
X_test = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\X_test_features.npy")
y_test = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\y_test.npy")

# Define the parameter grid based on the tables you provided
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 1.5]
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Initialize the classifiers
xgb = XGBClassifier(n_jobs=-1, random_state=42)
rf = RandomForestClassifier()
svm = SVC(probability=True)

grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)

# Fit the grids
grid_xgb.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)
grid_svm.fit(X_train, y_train)

# Best models
best_xgb = grid_xgb.best_estimator_
best_rf = grid_rf.best_estimator_
best_svm = grid_svm.best_estimator_

# Define the stacking ensemble
stack = StackingClassifier(estimators=[
    ('xgb', best_xgb),
    ('rf', best_rf),
    ('svm', best_svm)
], final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))

# Train the stacking classifier
stack.fit(X_train, y_train)


# Save the model to disk
filename = 'finalized_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(stack, file)

# Load the model from disk
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Evaluate loaded model
y_pred_loaded = loaded_model.predict(X_test)
print(f'Test Accuracy of loaded model: {accuracy_score(y_test, y_pred_loaded):.4f}')
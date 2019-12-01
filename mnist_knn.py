import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Downloading the data, making training and test sets.
mnist = fetch_openml('mnist_784', version=1)

X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiating the model, fitting it.
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Validating it.

print('Train Score', knn.score(X_train, y_train))
print('Test Score', knn.score(X_test, y_test))

print(knn.predict_proba(X_test)[:9])

param_grid = {
    'n_neighbors': [3,4, 5,7,9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
    }

gs = GridSearchCV(knn,
                  param_grid,
                  verbose=3,
                  cv=5
    )
gs_results = gs.fit(X_train, y_train)

print(gs_results.best_params_) # -- > Outputs the best model parameters
print(gs_results.best_score_) #  -- > Outputs the best score





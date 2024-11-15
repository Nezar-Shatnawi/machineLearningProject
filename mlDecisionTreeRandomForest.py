import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV , RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , precision_score ,recall_score , f1_score , classification_report
from sklearn.ensemble import RandomForestClassifier



"""
1. Loading the Data:
 Load the dataset using pandas.
 Inspect the dataset for structure, feature types, and missing values.
"""

df = pd.read_csv('covtype.csv')
"""
print(df.head())
print(df.describe())
print(df.dtypes)
# 2. Data Preprocessing

print(df.isnull().sum())
"""

"""
Data Splitting +
1. Decision Trees
2. RandomForest

"""
x = df.drop('Cover_Type' , axis=1)
y = df['Cover_Type']

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.3 , random_state = 42)

"""
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

tree_acc = classification_report(y_test,y_pred)
print(tree_acc)


print("--------------Hyperparameter Tuning DecisionTreeClassifier--------------------")

param_grid = {
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

print("Best Parameters GridSearchCV:", grid_search.best_params_)
print("Best Accuracy GridSearchCV:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(x_train)

best_grid_tree_acc = classification_report(y_test,y_pred)
print(best_grid_tree_acc)
"""



print("--------------RandomForest--------------------")

rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train,y_train)
y_pred_rfc = rf.predict(x_test)
rf_acc = classification_report(y_test,y_pred_rfc)
print(rf_acc)


print("--------------Hyperparameter Tuning RandomForestClassifier--------------------")

rf_model = RandomForestClassifier(random_state=42)

# define the parameter
param_dist = {
    'n_estimators': np.arange(50, 300, 50),       # number of trees in the forest
    'max_depth': np.arange(5, 20, 5),             # mnaximum depth of the trees
    'min_samples_split': [2, 5, 10],              # mninimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],                # mninimum number of samples required at a leaf node
    'bootstrap': [True, False]                    # whether to bootstrap samples when building trees
}

random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=100, cv=3,random_state=42, n_jobs=-1)

random_search.fit(x_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")

best_rf_model = random_search.best_estimator_
y_pred = best_rf_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
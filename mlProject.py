import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , precision_score ,recall_score , f1_score , classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


"""
1. Loading the Data:
 Load the dataset using pandas.
 Inspect the dataset for structure, feature types, and missing values.
"""

df = pd.read_csv('covtype.csv')

print(df.head())
print(df.describe())
print(df.dtypes)
# 2. Data Preprocessing

print(df.isnull().sum())


"""
Data Splitting +
Use Feature Scaling (StandardScaler) for:
Logistic Regression
k-Nearest Neighbors (k-NN)

"""
x = df.drop('Cover_Type' , axis=1)
y = df['Cover_Type']

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.3 , random_state = 42)

scaler_standard = StandardScaler().set_output(transform="pandas")
X_train_standard = scaler_standard.fit_transform(x_train)
X_test_standard = scaler_standard.transform(x_test)

# Apply Logistic Regression + k-Nearest Neighbors
models = {
    'logreg' : LogisticRegression(max_iter=500) ,
    'knn' : KNeighborsClassifier()
}

results_list = []

for model_name, model in models.items():

        model.fit(X_train_standard, y_train)
        y_pred = model.predict(X_test_standard)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred , average='macro')
        prec = precision_score(y_test, y_pred , average='macro')
        f_score = f1_score(y_test, y_pred , average='macro')


        results_list.append({
            "Model": model_name,
            "Acc": accuracy,
            "recall" : recall ,
            "precision" :prec ,
            "f1_score":f_score
        })


results = pd.DataFrame(results_list)

print(results)

print("---------------------------------------")

# GridSearchCV for Logistic Regression
"""
log_param_grid = {
    'C': [0.1, 0.5, 1.0, 5.0],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'max_iter': [200 , 300, 400],  # More options for max_iter
    'penalty' : ['l1' , 'l2']
}


log_reg = LogisticRegression()

grid_search = GridSearchCV(estimator=log_reg, param_grid=log_param_grid, cv=3, scoring='accuracy')

grid_search.fit(X_train_standard, y_train)

print("Best Parameters GridSearchCV:", grid_search.best_params_)
print("Best Accuracy GridSearchCV:", grid_search.best_score_)


best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_standard)

scores = {
    "Accuracy": accuracy_score(y_test, y_pred_best),
    "Recall": recall_score(y_test, y_pred_best, average='micro'),
    "Precision": precision_score(y_test, y_pred_best, average='micro'),
    "F1 Score": f1_score(y_test, y_pred_best, average='micro')
}

res_list = []

for score_name, score_value in scores.items():
    res_list.append({
        "Score Name": score_name,
        "Score Value": score_value
    })

res = pd.DataFrame(res_list)
print(res)
"""

print("---------------------------------------")

knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Example values for number of neighbors
    'weights': ['uniform', 'distance'],  # Weighting function
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
}
knn_model = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10)
knn_model.fit(x_train, y_train)

print("Best Parameters:", knn_model.best_params_)
print("Best Cross-Validation Score:", knn_model.best_score_)

best_knn = knn_model.best_estimator_
y_pred = best_knn.predict(x_test)


print(classification_report(y_test, y_pred))
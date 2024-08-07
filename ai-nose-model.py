import pandas as pd
import numpy as np
import os
from time import sleep

# import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

TRAIN_PATH = "train"
TEST_PATH = "test"
TEST_ORIGIN = "test_single"

trainFilePath = os.path.abspath(TRAIN_PATH)
testFilePath = os.path.abspath(TEST_PATH)

# Train data
trainData = pd.read_csv(os.path.join(trainFilePath, "train.csv"))
trainData = trainData.iloc[:, :-2]
trainData["category"] = trainData["category"].astype("category")

# Test data
# testData = pd.read_csv(os.path.join(testFilePath, "test.csv"))
# testData = testData.iloc[:, :-2]

y = trainData["category"]
X = trainData.drop("category", axis=1)

# category = {"coffee": 0, "kahlua":1, "lrishCream":2, "rum":3}
# y_trans = y.map(category)
# print(y_trans)

# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = GradientBoostingClassifier()
rfc = RandomForestClassifier(random_state=42)

stand = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train = stand.fit_transform(X_train)
X_test = stand.transform(X_test)

# ! Get best parameters
# param_grid = {
#     "n_estimators": [2, 5, 8, 10, 15, 20, 30],
#     "max_features": ["sqrt", "log2"],
#     "max_depth": [4, 5, 6, 7, 8, 9, 10],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 5],
#     # "criterion": ["gini", "entropy"],
#     "bootstrap": [True, False],
# }

# CV_rfc = GridSearchCV(
#     estimator=rfc, param_grid=param_grid, error_score="raise", cv=5, n_jobs=1
# )

# CV_rfc = RandomizedSearchCV(
#     estimator=rfc,
#     param_distributions=param_grid,
#     n_iter=100,
#     cv=5,
#     n_jobs=-1,
#     verbose=1,
#     random_state=42,
# )

# CV_rfc.fit(X_train, y_train)
# best_params = CV_rfc.best_params_
# print(best_params)

# Best params GridSearchCV:
# {'criterion': 'gini', 'max_depth': 9, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 30}

# Best params RandomSearchCV:
# {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 8, 'criterion': 'entropy', 'bootstrap': False}

# Model
model = RandomForestClassifier(
    # criterion="entropy",
    max_depth=10,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=15,
    bootstrap=False,
)

# ! Accuracy evaluation
model.fit(X_train, y_train)

# Get the training score
train_score = model.score(X_train, y_train)
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Score: ", train_score)
print("Training Accuracy: ", train_accuracy)

# Get the test score
test_score = model.score(X_test, y_test)
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Score: ", test_score)
print("Test Accuracy: ", test_accuracy)

y_pred = model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# ! Test Data
# def prediction_realtime(odor):
#     # print("input : ", odor)
#     # odor = odor.split(";")[1:-2]
#     odor = np.array(odor)
#     odor = pd.DataFrame([odor], columns=trainData.columns[1:])
#     odor = stand.transform(odor)
#     y_pred = model.predict(odor)
#     print("\nOutput : ", y_pred[0], "\n")


# testFile = os.listdir(TEST_ORIGIN)

# X = stand.fit_transform(X)
# # testData = stand.transform(testData)
# model.fit(X, y)

# i = 0
# while i < len(testFile):
#     testFilePath = os.path.abspath(TEST_ORIGIN + "/" + testFile[i])
#     testData = pd.read_csv(os.path.join(testFilePath))
#     print("Input: ", testData.columns[0])
#     test = testData.columns[0].split(";")[1:-2]
#     # print(test)
#     prediction_realtime(test)
#     i += 1
#     sleep(1)

# y_pred = model.predict(testData)
# test = pd.DataFrame([y_pred])
# print(y_pred)

# ! CLI test odor
# def prediction():
#     odor = input("input : ")
#     odor = odor.split(";")[1:-2]
#     odor = np.array(odor)
#     odor = pd.DataFrame([odor], columns=trainData.columns[1:])
#     odor = stand.transform(odor)
#     y_pred = model.predict(odor)
#     print("\nOutput : ", y_pred[0], "\n")

# while True:
#     prediction()
#     sleep(1)

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATASET_TRAIN_FILE = 'datasets\\ds.csv'

df_train = pd.read_csv(DATASET_TRAIN_FILE)

x = df_train.iloc[:, :-1].values
y = df_train.iloc[:, 23].values


def knn():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    clf = KNeighborsClassifier(n_neighbors=25)
    clf.fit(x_train, y_train)
    return clf, x_test, y_test, y_train


def decision_tree():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    return clf, x_test, y_test, y_train


def random_forest(n_est, max_dpth):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_dpth)
    clf.fit(x_train, y_train)
    return clf, x_test, y_test, y_train


x_points = []
y_points = []
for repeats in range(0, 100):
    best_accuracy = [0]
    for i in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]:
        for j in range(1, 33):
            ALGORITHM = random_forest(i, j)

            classifier, X_test, Y_test, Y_train = ALGORITHM
            Y_pred = classifier.predict(X_test)
            # report = classification_report(Y_test, Y_pred)
            # matrix = confusion_matrix(Y_test, Y_pred)
            accuracy = accuracy_score(Y_test, Y_pred)
            if accuracy > best_accuracy[0]:
                best_accuracy = [accuracy, i, j]
    x_points.append(best_accuracy[1])
    y_points.append(best_accuracy[2])

plt.scatter(x_points, y_points)
plt.axis([0, 200, 0, 32])
plt.xlabel('n estimators')
plt.ylabel('max depth')
plt.show()


def print_results():
    print("Accuracy: " + str(accuracy) + "\n")
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))

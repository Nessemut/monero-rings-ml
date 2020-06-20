import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


DATASET_TRAIN_FILE = 'datasets\\ds.csv'

df_train = pd.read_csv(DATASET_TRAIN_FILE)

x = df_train.iloc[:, :-1].values
y = df_train.iloc[:, 23].values

def knn():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    clf = KNeighborsClassifier(n_neighbors = 25)
    clf.fit(x_train, y_train)
    return clf, x_test, y_test, y_train

def decision_tree():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    return clf, x_test, y_test, y_train

def random_forest():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(x_train, y_train)
    return clf, x_test, y_test, y_train

ALGORITHM = random_forest()

classifier, X_test, Y_test, Y_train = ALGORITHM
Y_pred = classifier.predict(X_test)
report = classification_report(Y_test, Y_pred)
matrix = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)


print("Accuracy: " + str(accuracy) + "\n")
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

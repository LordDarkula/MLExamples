import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

X_train = np.array([[0, 0],
                    [2, 3],
                    [7, 6],
                    [8, 8],
                    [6, 9]])

y_train = np.array([0, 0, 1, 1, 1])

X_test = np.array([[9, 9], [-1, -1]])
y_test = np.array([1, 0])

clf = svm.SVC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print classification_report(y_test, y_pred)

import numpy as np
import pandas as pd

from tools import split_data

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.savefig("images/SVM_feature_importance.png")


def svm_fn(X_train, X_val, X_test, y_train, y_val, y_test):

    # SVM
    clf_svm = svm.SVC(kernel="linear", gamma='scale')
    clf_svm.fit(X_train, y_train)

    train_acc_svm = clf_svm.score(X_train, y_train)
    test_acc_svm = clf_svm.score(X_val, y_val)
    print("Train:", train_acc_svm)
    print("Test:", test_acc_svm)

    test_pred_svm = clf_svm.predict(X_val)

    print(classification_report(y_val, test_pred_svm,
                                target_names=["Non-Scam", "Ponzi"], digits=4))

    disp_svm = ConfusionMatrixDisplay(confusion_matrix(
        y_val, test_pred_svm), display_labels=["Non-Scam", "Ponzi"])
    disp_svm.plot()
    plt.title("SVM Classifier: Confusion Matrix (Validation)")
    plt.savefig("images/SVM_confusion_matrix_val.png")

    y_pred = clf_svm.predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Scam", "Ponzi"], digits=4))
    disp_rf = ConfusionMatrixDisplay(confusion_matrix(
        y_test, y_pred), display_labels=["Non-Scam", "Ponzi"])
    disp_rf.plot()
    plt.title("Random Forest Classifier: Confusion Matrix (Test)")
    plt.savefig("images/SVM_confusion_matrix_val.png")

    return clf_svm


if __name__ == "__main__":
    data = pd.read_csv("all_data.csv")

    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1:]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    clf_svm = svm_fn(X_train, X_val, X_test, y_train, y_val, y_test)

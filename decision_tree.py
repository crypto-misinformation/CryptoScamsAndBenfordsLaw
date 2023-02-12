import numpy as np
import pandas as pd

from tools import split_data, feature_labels

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def dt(X_train, X_val, X_test, y_train, y_val, y_test):
    # Decision Tree (with AdaBoost)
    clf_dt = tree.DecisionTreeClassifier(
        criterion='entropy', splitter="best", min_samples_split=2, min_samples_leaf=1)
    clf_dt.fit(X_train, y_train)

    train_acc_dt = clf_dt.score(X_train, y_train)
    test_acc_dt = clf_dt.score(X_val, y_val)
    print("Train:", train_acc_dt)
    print("Test:", test_acc_dt)

    test_pred_dt = clf_dt.predict(X_val)

    print(classification_report(y_val, test_pred_dt,
                                target_names=["Non-Scam", "Ponzi"], digits=4))

    disp_svm = ConfusionMatrixDisplay(confusion_matrix(
        y_val, test_pred_dt), display_labels=["Non-Scam", "Ponzi"])
    disp_svm.plot()
    plt.title("Decision Tree Classifier: Confusion Matrix")
    plt.savefig("images/DT_confusion_matrix_val.png")

    y_pred = clf_dt.predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Scam", "Ponzi"], digits=4))
    disp_rf = ConfusionMatrixDisplay(confusion_matrix(
        y_test, y_pred), display_labels=["Non-Scam", "Ponzi"])
    disp_rf.plot()
    plt.title("Decision Tree Classifier: Confusion Matrix (Test)")
    plt.savefig("images/DT_confusion_matrix_test.png")

    feat_importance_dt = clf_dt.tree_.compute_feature_importances(
        normalize=False)
    plt.clf()
    plt.figure(figsize=(6, 9))
    plt.xlim((0, .14))
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.title("Feature Importance: Decision tree")
    plt.barh([x for x in range(len(feat_importance_dt))], feat_importance_dt)
    for i, v in enumerate(feat_importance_dt):
        plt.text(v+0.0005, i, str(feature_labels[i+1]),
                 color='black', fontweight='bold')

    plt.savefig("images/DT_feature_importance.png")
    print(feat_importance_dt)

    return clf_dt


if __name__ == "__main__":
    data = pd.read_csv("all_data.csv")

    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1:]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    clf_dt = dt(X_train, X_val, X_test, y_train, y_val, y_test)

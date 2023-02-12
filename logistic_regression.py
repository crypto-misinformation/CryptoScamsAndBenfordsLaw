import numpy as np
import pandas as pd

from tools import split_data

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def lr(X_train, X_val, X_test, y_train, y_val, y_test):
    clf_lr = LogisticRegression(
        max_iter=1000, penalty='l2', solver='lbfgs').fit(X_train, y_train)

    feat_imp_lr = clf_lr.coef_[0]

    test_pred_lr = clf_lr.predict(X_val)

    # Only Using train and validation set
    train_acc_lr = clf_lr.score(X_train, y_train)
    test_acc_lr = clf_lr.score(X_val, y_val)
    print("Train:", train_acc_lr)
    print("Test:", test_acc_lr)

    print(classification_report(y_val, test_pred_lr,
                                target_names=["Non-Scam", "Ponzi"], digits=4))

    disp_lr = ConfusionMatrixDisplay(confusion_matrix(
        y_val, test_pred_lr), display_labels=["Non-Scam", "Ponzi"])
    disp_lr.plot()
    plt.title("Logistic Regression Classifier: Confusion Matrix")
    plt.savefig("images/LR_confusion_matrix_val.png")

    # Done Training -- prediction on test set
    y_pred = clf_lr.predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Scam", "Ponzi"], digits=4))

    disp_rf = ConfusionMatrixDisplay(confusion_matrix(
        y_test, y_pred), display_labels=["Non-Scam", "Ponzi"])
    disp_rf.plot()
    plt.title("Logistic Regression Classifier: Confusion Matrix (test)")
    plt.savefig("images/LR_confusion_matrix_test.png")

    plt.clf()
    plt.title("Feature Importance: Logistic Regression")
    plt.bar([x for x in range(len(feat_imp_lr))], feat_imp_lr)
    plt.savefig("images/LR_feature_importance.png")

    return clf_lr


if __name__ == "__main__":
    data = pd.read_csv("all_data.csv")

    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1:]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    clf_lr = lr(X_train, X_val, X_test, y_train, y_val, y_test)

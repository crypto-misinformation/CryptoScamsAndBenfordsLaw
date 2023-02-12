import numpy as np
import pandas as pd

from tools import split_data

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def rf(X_train, X_val, X_test, y_train, y_val, y_test):

    # Random Forest
    clf_rf = RandomForestClassifier(max_depth=13)
    clf_rf.fit(X_train, y_train)

    feat_imp_rf = clf_rf.feature_importances_

    # Only Using train and validation set
    train_acc_rf = clf_rf.score(X_train, y_train)
    test_acc_rf = clf_rf.score(X_val, y_val)
    print("Train:", train_acc_rf)
    print("Test:", test_acc_rf)

    test_pred_rf = clf_rf.predict(X_val)

    print(classification_report(y_val, test_pred_rf,
                                target_names=["Non-Scam", "Ponzi"], digits=4))

    disp_rf = ConfusionMatrixDisplay(confusion_matrix(
        y_val, test_pred_rf), display_labels=["Non-Scam", "Ponzi"])
    disp_rf.plot()
    plt.title("Random Forest Classifier: Confusion Matrix (validation)")
    plt.savefig("images/RF_confusion_matrix_val.png")

    # Done Training -- prediction on test set
    y_pred = clf_rf.predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Scam", "Ponzi"], digits=4))
    disp_rf = ConfusionMatrixDisplay(confusion_matrix(
        y_test, y_pred), display_labels=["Non-Scam", "Ponzi"])
    disp_rf.plot()
    plt.title("Random Forest Classifier: Confusion Matrix (test)")
    plt.savefig("images/RF_confusion_matrix_test.png")

    plt.clf()
    plt.title("Feature Importance: Random Forest")
    plt.bar([x for x in range(len(feat_imp_rf))], feat_imp_rf)
    plt.savefig("images/RF_feature_importance.png")

    return clf_rf


if __name__ == "__main__":
    data = pd.read_csv("all_data.csv")

    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1:]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    clf_rf = rf(X_train, X_val, X_test, y_train, y_val, y_test)

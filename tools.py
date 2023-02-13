import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

feature_labels = ["Address", "in_count", "out_count", "in_unique", "out_unique", "total_count", "in_gas_limit_avg", "in_gas_limit_std", "in_gas_limit_med",
                  "out_gas_limit_avg", "out_freq_std", "out_gas_limit_med", "in_value_avg", "in_value_std", "out_value_avg",
                  "out_value_std", "benford_chi_sq_1", "benford_kstest_1", "benford_chi_sq_2", "benford_kstest_2", "class"]


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=29)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_test = np.ravel(y_test)
    y_train = np.ravel(y_train)
    y_val = np.ravel(y_val)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def calc_true_benford_values():
    """returns the distribution for Benford's Law for first, 
    second and third digits

    Returns:
        list, list, list: benfords distribution for first, second and 
        third digits
    """
    benford_first = []
    benford_second = []
    benford_third = []

    for i in range(0, 10):

        # First digit calculation
        if i != 0:
            bf = np.log10(1 + 1/i)
            benford_first.append(bf)

        # Second Digit Calculation
        bs = 0
        for j in range(1, 10):
            bs += np.log10(1 + 1/(i+(j*10)))
        benford_second.append(bs)

    return benford_first, benford_second

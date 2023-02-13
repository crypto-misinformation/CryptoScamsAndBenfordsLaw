import numpy as np
import pandas as pd
import os
import re
from scipy.stats import chisquare, kstest
from tools import calc_true_benford_values


feature_labels = ["Address", "in_count", "out_count", "in_unique", "out_unique", "total_count", "in_gas_limit_avg", "in_gas_limit_std", "in_gas_limit_med",
                  "out_gas_limit_avg", "out_freq_std", "out_gas_limit_med", "in_value_avg", "in_value_std", "out_value_avg",
                  "out_value_std", "benford_chi_sq_1", "benford_kstest_1", "benford_chi_sq_2", "benford_kstest_2", "class"]


def num_txs_data(df, addr):
    # returns incoming txs, outgoing txs, and total txs

    incoming_txs = df[df["to"] == addr]
    outgoing_txs = df[df["from"] == addr]

    return len(incoming_txs), len(outgoing_txs), df.shape[0]


def outgoing_df(df, addr):
    outgoing_addrs = df[df["from"] == addr]
    return outgoing_addrs


def incoming_df(df, addr):
    incoming_addrs = df[df["to"] == addr]
    return incoming_addrs


def avg_incoming_tx_freq(df, addr):
    # Return average and std and median
    incoming_txs = df[df["to"] == addr]
    incoming_txs.sort_values('blockNumber')
    bn = incoming_txs["blockNumber"]

    bn = pd.to_numeric(bn)
    if bn.shape[0] == 1:
        return 0, 0, 0

    diffs = [t - s for s,
             t in zip(bn, bn[1:])]
    return np.average(diffs), np.std(diffs), np.median(diffs)


def avg_outgoing_tx_freq(df, addr):
    # Return average and std and median
    outgoing_txs = df[df["from"] == addr]
    outgoing_txs.sort_values('blockNumber')
    bn = outgoing_txs["blockNumber"]

    bn = pd.to_numeric(bn)
    if bn.shape[0] == 1:
        return 0, 0, 0

    diffs = [t - s for s,
             t in zip(bn, bn[1:])]

    return np.average(diffs), np.std(diffs), np.median(diffs)


def outgoing_addrs(df, addr):
    outgoing_addrs = df[df["from"] == addr]["to"]
    return list(outgoing_addrs)


def incoming_addrs(df, addr):
    incoming_addrs = df[df["to"] == addr]["from"]
    return list(incoming_addrs)


def avg_incoming_gas_limit(df, addr):
    # Return average and std
    in_gas_limit = df[df["to"] == addr]["gasLimit"]
    in_gas_limit = pd.to_numeric(in_gas_limit)

    return np.average(in_gas_limit), np.std(in_gas_limit), np.median(in_gas_limit)


def avg_outgoing_gas_limit(df, addr):
    # Return average and std
    out_gas_limit = df[df["from"] == addr]["gasLimit"]
    out_gas_limit = pd.to_numeric(out_gas_limit)

    return np.average(out_gas_limit), np.std(out_gas_limit), np.median(out_gas_limit)


def avg_in_tx_value(df, addr):
    in_tx_values = df[df["to"] == addr]["value"].astype(float)
    #in_tx_values = pd.to_numeric(in_tx_values)
    return np.average(in_tx_values), np.std(in_tx_values)


def avg_out_tx_value(df, addr):
    out_tx_values = df[df["from"] == addr]["value"].astype(float)

    #out_tx_values = pd.to_numeric(out_tx_values)
    return np.average(out_tx_values), np.std(out_tx_values)


def unique_address_counts(df, addr):
    o = set(outgoing_addrs(df, addr))
    i = set(incoming_addrs(df, addr))

    return len(o), len(i)


def benfords_metrics(df):
    nums = list(df["value"])

    tx_amounts = [value for value in nums if float(value) != 0.0]

    start_nums = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    second_nums = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    nums_ponzi = []
    for t in tx_amounts:
        if float(t) == 0:
            continue
        i = 0
        s = str(t)
        while (s[i] == '.' or s[i] == '0') and i < len(str(t))-2:
            i += 1

        key = int(s[i])

        if len(s) == 1 or s[i+1] == "e":
            key2 = 0
        elif s[i+1] == ".":
            key2 = int(str(t)[i+2])
        else:
            key2 = int(str(t)[i+1])

        if key != 0:
            start_nums[key] = start_nums[key] + 1
        nums_ponzi.append(key)
        second_nums[key2] = second_nums[key2] + 1

    total = 0
    second_total = second_nums[0]
    for i in range(1, 10):
        total += start_nums[i]
        second_total += second_nums[i]

    if total == 0:
        return np.NaN

    decimal = []
    second_nums_decimal = []

    benford, benford_second = calc_true_benford_values()

    second_nums_decimal.append(second_nums[0]/second_total)
    for k in range(1, 10):
        decimal.append(start_nums[k]/total)
        second_nums_decimal.append(second_nums[k]/second_total)

    cs = chisquare(decimal, benford)
    cs_s = cs.statistic
    cs_p = cs.pvalue
    ks = kstest(decimal, benford, mode="exact")
    ks_stat = ks.statistic
    ks_p = ks.pvalue

    first = [float(cs_s), float(cs_p), float(ks_stat), float(ks_p)]

    cs = chisquare(second_nums_decimal, benford_second)
    cs_s = cs.statistic
    cs_p = cs.pvalue
    ks = kstest(second_nums_decimal, benford_second, mode="exact")
    ks_stat = ks.statistic
    ks_p = ks.pvalue
    second = [float(cs_s), float(cs_p), float(ks_stat), float(ks_p)]

    return first, second


def get_features(df, addr, c=0):
    num_incoming, num_outgoing, total_num_txs = num_txs_data(df, addr)

    in_freq_avg, in_freq_std, in_freq_med = avg_incoming_tx_freq(df, addr)
    out_freq_avg, out_freq_std, out_freq_med = avg_outgoing_tx_freq(df, addr)

    in_gas_limit_avg, in_gas_limit_std, in_gas_limit_med = avg_incoming_gas_limit(
        df, addr)
    out_gas_limit_avg, out_gas_limit_std, out_gas_limit_med = avg_incoming_gas_limit(
        df, addr)

    in_tx_val_avg, in_tx_val_std = avg_in_tx_value(df, addr)
    out_tx_val_avg, out_tx_val_std = avg_out_tx_value(df, addr)

    in_unique, out_unique = unique_address_counts(df, addr)

    first, second = benfords_metrics(df)

    cs_first = first[0]
    ks_stat_first = first[1]

    cs_second = second[0]
    ks_stat_second = second[1]

    feature_vec = [addr, num_incoming, num_outgoing, in_unique, out_unique, total_num_txs,
                   in_gas_limit_avg, in_gas_limit_std, in_gas_limit_med,
                   out_gas_limit_avg, out_gas_limit_std, out_gas_limit_med, in_tx_val_avg, in_tx_val_std, out_tx_val_avg,
                   out_tx_val_std, cs_first, ks_stat_first, cs_second, ks_stat_second, c]

    return pd.DataFrame(feature_vec)


def create_address_dataframe(dir, c=0):
    data_files = os.listdir(dir)

    all = pd.DataFrame(columns=feature_labels)

    for d in data_files:
        df = pd.read_csv(dir+d)
        df = df.sort_values("blockNumber")
        a = re.split("\_|\.", d)[1]

        try:
            feats = get_features(df, a, c)

            if all.shape[0] == 0:
                all = feats.T
            else:
                all = pd.concat([all, feats.T], axis=0, ignore_index=True)
        except:
            print("Error with", d)

    all.columns = feature_labels
    return all


if __name__ == "__main__":
    # Assumes that address data is separated into csv files by user
    # in directories "data/by_address/legit/" and "data/by_address/ponzi/"

    ponzi = create_address_dataframe("data/by_address/ponzi/", 1)
    legit = create_address_dataframe("data/by_address/legit/", 0)

    all_data = pd.concat([legit, ponzi], ignore_index=True)
    final = all_data.dropna()               # cleanup
    final = final.drop_duplicates()
    final.to_csv("all_data_unsplit.csv", index=False)

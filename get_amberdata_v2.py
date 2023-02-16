from constants import AMBER_KEY
import json
import pandas as pd
import requests
import os


def extract_Addr(ad):
    if type(ad) == list:
        return ad[0]["address"]
    return ad["address"]


def normalize_df(df):
    if df.shape[0] == 0:
        return df

    df['from'] = df['from'].apply(extract_Addr)
    df['to'] = df['to'].apply(extract_Addr)
    return df


def get_internal_transactions_df(addr):
    """Deprecated
    """
    url = "https://web3api.io/api/v2/addresses/{0}/functions".format(addr)

    headers = {
        "Accept": "application/json",
        "x-amberdata-blockchain-id": "ethereum-mainnet",
        "x-api-key": AMBER_KEY
    }

    all_data = pd.DataFrame()

    i = 0
    while all_data.shape[0] % 100 == 0 and all_data.shape[0] < 10000:
        querystring = {"page": str(i), "size": "100",
                       "includeTokenTransfers": "true"}

        response = requests.request(
            "GET", url, headers=headers, params=querystring)
        r = json.loads(response.text)

        i += 1
        if r["status"] == 200:
            payload = r["payload"]["records"]
            txData = pd.DataFrame(payload)

            all_data = pd.concat((all_data, txData))

            if all_data.shape[0] == 0:
                break

        else:
            break

    all_data = all_data.rename(columns={"initialGas": "gasLimit"})

    return normalize_df(all_data)


def get_external_transactions_df(addr):
    url = "https://web3api.io/api/v2/addresses/{0}/transactions".format(addr)

    headers = {
        "Accept": "application/json",
        "x-amberdata-blockchain-id": "ethereum-mainnet",
        "x-api-key": AMBER_KEY
    }

    all_data = pd.DataFrame()

    i = 0
    # Hard cut bc there are a couple of smart contracts in here
    while all_data.shape[0] % 100 == 0 and all_data.shape[0] < 100000:
        querystring = {"includeFunctions": "true", "decodeTransactions": "false",
                       "includeTokenTransfers": "true", "page": str(i), "size": "100"}

        response = requests.request(
            "GET", url, headers=headers, params=querystring)
        r = json.loads(response.text)

        i += 1
        if r["status"] == 200:
            payload = r["payload"]["records"]
            txData = pd.DataFrame(payload)

            all_data = pd.concat((all_data, txData))

            if all_data.shape[0] == 0:
                break

        else:
            break
    return normalize_df(all_data)


def pull_token_transfers(df):
    if "tokenTransfers" not in df.columns:
        return pd.DataFrame()

    all = pd.DataFrame()

    for index, row in df.iterrows():
        if type(row["tokenTransfers"]) == list:
            d = pd.DataFrame(row["tokenTransfers"])
            d["blockNumber"] = row["blockNumber"]
            all = pd.concat([all, d])

    all = all.rename(columns={"amount": "value"})

    return all


def pull_function_transfers(df):
    if "functions" not in df.columns:
        return pd.DataFrame()
    all = pd.DataFrame()

    for index, row in df.iterrows():
        if type(row["functions"]) == list:
            d = pd.DataFrame(row["functions"])
            d["blockNumber"] = row["blockNumber"]
            d["gasLimit"] = row["gasLimit"]
            d = d.loc[d['value'] != '0']
            all = pd.concat([all, d])

    na = normalize_df(all)

    if na.shape[0] == 0:
        return pd.DataFrame()
    return na


def get_all_transactions(addr):
    external = get_external_transactions_df(addr)

    token_transfers = pull_token_transfers(external)
    functions = pull_function_transfers(external)

    complete = pd.concat(
        [external, token_transfers, functions], ignore_index=True)
    return complete


def compile_transactions_df(addrs, t="legit", save_csv=True):
    data = pd.DataFrame()

    for i, a in enumerate(addrs):
        name = "data/by_address/"+t+"/"+t+"_" + a + ".csv"

        if os.path.exists(name) == False:
            print("(UPDATE)\t Reading Address", a)
            try:
                df = get_all_transactions(a)
                df = df[["blockNumber", "from", "to", "gasLimit", "value"]]

                if save_csv:
                    df.to_csv(name)
                print("(FINISHED)\t Wrote", name)
                if data.shape[0] == 0:
                    data = df
                elif df.shape[0] != 0:
                    data = pd.concat((data, df))

            except:
                print("(ERROR)\t\t Address:", a)

        else:
            print("(UPDATE)\t File already exists for", a)

        if i % 25 == 0:
            print("(PROGRESS)\t Address {0}/{1}".format(i+1, len(addrs)+1))

    return data


if __name__ == "__main__":

    with open("data/addresses_legit.txt") as f:
        addrs = f.read().split()

    # Note: if you need to save computer space you can change it to not save csv for each address
    # This will required you to adjust the feature calculation in the gen_feature_matrix file.

    # t should be legit or ponzi
    df = compile_transactions_df(list(set(addrs)), t="legit", save_csv=True)

    # optional if you choose not to save csv files
    df.to_csv("legit_transactions_all.csv")

    # IF YOU RUN IT AND GET ALL ERRORS, CHECK AMBERDATA KEY IN CONSTANTS
    # IF YOU NEED TO REQUEST AN AMBERDATA KEY, BEST METHOD IS TO REACH OUT ON TWITTER
    # OR EMAIL

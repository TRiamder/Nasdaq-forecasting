import sqlite3
import pandas as pd
from src.functions.func_mu_inf import mutual_information

def mutual_information_pre():
    # set up data
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query("SELECT * FROM nasdaq_macros_added", conn)

    df.set_index('date', inplace=True)

    df_train = df.loc[:'2021-12-31']

    y_high = df_train["high"]
    y_low = df_train["low"]

    target = ["high", "low"]
    X = df_train.drop(target, axis=1)

    # create mutual information scores separately for high and low
    mi_high, mi_low = mutual_information(X, y_high, y_low)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    print('Mutual Information Score for High Target:')
    print(mi_high)
    print('Mutual Information Score for Low Target:')
    print(mi_low)

if __name__ == '__main__':
    mutual_information_pre()
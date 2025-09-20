import sqlite3
import pandas as pd
from src.functions.func_mu_inf import mutual_information

def mutual_information_res():
    # set up the data
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        X_train = pd.read_sql_query('SELECT * FROM X_train', conn)
        y_train_high_res = pd.read_sql_query('SELECT * FROM y_train_high_res', conn)
        y_train_low_res = pd.read_sql_query('SELECT * FROM y_train_low_res', conn)

    X_train.set_index('date', inplace=True)
    y_train_high_res.set_index('date', inplace=True)
    y_train_low_res.set_index('date', inplace=True)

    # create and print mutual information scores
    mi_high, mi_low = mutual_information(X_train, y_train_high_res.high, y_train_low_res.low)
    pd.set_option('display.max_columns', 150)
    pd.set_option('display.max_rows', 150)
    print('Mutual Information Score for High Residuals:')
    print(mi_high)
    print("--" * 40)
    print('Mutual Information Score for Low Residuals:')
    print(mi_low)

if __name__ == '__main__':
    mutual_information_res()
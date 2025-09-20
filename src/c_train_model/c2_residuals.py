import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def residuals():
    # set up the data
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query('SELECT * FROM nasdaq_macros_added', conn)
        y_train_pred = pd.read_sql_query('SELECT * FROM train_predictions', conn)
        y_val_pred = pd.read_sql_query('SELECT * FROM val_predictions', conn)
        y_test_pred = pd.read_sql_query('SELECT * FROM test_predictions', conn)

    df.set_index('date', inplace=True)
    y_train_pred.set_index('date', inplace=True)
    y_val_pred.set_index('date', inplace=True)
    y_test_pred.set_index('date', inplace=True)

    train_data = df.loc[:'2021-12-31']
    val_data = df.loc['2022-01-01':'2023-12-31']
    test_data = df.loc['2024-01-01':]

    targets = ["high", "low"]

    y_train = train_data[targets]
    y_val = val_data[targets]
    y_test = test_data[targets]

    # create residuals by subtracting the predictions of the linear regression model from the real target values
    y_train_high_res = y_train["high"] - y_train_pred["high"]
    y_train_low_res = y_train["low"] - y_train_pred["low"]
    y_val_high_res = y_val["high"] - y_val_pred["high"]
    y_val_low_res = y_val["low"] - y_val_pred["low"]
    y_test_high_res = y_test["high"] - y_test_pred["high"]
    y_test_low_res = y_test["low"] - y_test_pred["low"]

    X_train = train_data.drop(targets, axis=1)
    X_val = val_data.drop(targets, axis=1)
    X_test = test_data.drop(targets, axis=1)

    # plot the values of the residuals
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    date_parsed = pd.to_datetime(y_train_high_res.index, format='%Y-%m-%d %H:%M:%S')
    sns.lineplot(x=date_parsed, y=y_train_high_res, ax=axs[0])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Residual Price Daily High')
    axs[0].set_title("High Residuals")
    sns.lineplot(x=date_parsed, y=y_train_low_res, ax=axs[1])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Residual Price Daily Low')
    axs[1].set_title("Low Residuals")
    plt.savefig('images/residuals.png')
    plt.show()

    # create lags of the residuals
    y_high_res = pd.concat([y_train_high_res, y_val_high_res, y_test_high_res], axis=0)
    y_low_res = pd.concat([y_train_low_res, y_val_low_res, y_test_low_res], axis=0)

    for shifts in range(1, 20):
        high_res_lag = y_high_res.shift(shifts)
        low_res_lag = y_low_res.shift(shifts)

        X_train[f"high_res_lag{shifts}"] = high_res_lag.loc[:'2021-12-31']
        X_train[f"low_res_lag{shifts}"] = low_res_lag.loc[:'2021-12-31']
        X_val[f"high_res_lag{shifts}"] = high_res_lag.loc['2022-01-01':'2023-12-31']
        X_val[f"low_res_lag{shifts}"] = low_res_lag.loc['2022-01-01':'2023-12-31']
        X_test[f"high_res_lag{shifts}"] = high_res_lag.loc['2024-01-01':]
        X_test[f"low_res_lag{shifts}"] = low_res_lag.loc['2024-01-01':]

    X_train.dropna(inplace=True)

    # align the targets from the training data with their respective features
    # since creating lags sets the first values in the features table to NA
    y_train_high_res = y_train_high_res.loc[X_train.index]
    y_train_low_res = y_train_low_res.loc[X_train.index]

    #print(X_train.columns)
    #print(X_val.columns)
    #print(X_test.columns)

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        X_train.to_sql('X_train', conn, if_exists='replace')
        X_val.to_sql('X_val', conn, if_exists='replace')
        X_test.to_sql('X_test', conn, if_exists='replace')
        y_train_high_res.to_sql('y_train_high_res', conn, if_exists='replace')
        y_train_low_res.to_sql('y_train_low_res', conn, if_exists='replace')
        y_val_high_res.to_sql('y_val_high_res', conn, if_exists='replace')
        y_val_low_res.to_sql('y_val_low_res', conn, if_exists='replace')
        y_test_high_res.to_sql('y_test_high_res', conn, if_exists='replace')
        y_test_low_res.to_sql('y_test_low_res', conn, if_exists='replace')

if __name__ == '__main__':
    residuals()
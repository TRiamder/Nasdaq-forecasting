import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

def pacf_lags():
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        X_train = pd.read_sql_query('SELECT * FROM X_train', conn)

    # create pacf plots for the lagged features
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    ax = ax.flatten()
    plot_pacf(X_train.volume_1, lags=20, ax=ax[0])
    ax[0].set_title("volume_1")
    plot_pacf(X_train.return_pct_lag1, lags=20, ax=ax[1])
    ax[1].set_title("return_pct_lag1")
    plot_pacf(X_train.range_lag1, lags=20, ax=ax[2])
    ax[2].set_title("range_lag1")
    plot_pacf(X_train.high_res_lag1, lags=20, ax=ax[4])
    ax[4].set_title("high_res_lag1")
    plot_pacf(X_train.low_res_lag1, lags=20, ax=ax[5])
    ax[5].set_title("low_res_lag1")
    fig.delaxes(ax[3])
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[4].grid(True)
    ax[5].grid(True)
    fig.suptitle("PACF Plots for Lag Features")
    plt.savefig('images/pacf_lags.png')
    plt.show()

if __name__ == '__main__':
    pacf_lags()
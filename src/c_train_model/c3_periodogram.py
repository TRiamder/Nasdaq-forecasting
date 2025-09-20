import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram

def periodogram_lags():
    # set up data
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        y_train_high_res = pd.read_sql_query("SELECT * FROM y_train_high_res", conn)
        y_train_low_res = pd.read_sql_query("SELECT * FROM y_train_low_res", conn)

    y_train_high_res.set_index('date', inplace=True)
    y_train_low_res.set_index('date', inplace=True)

    # create and plot periodograms
    freqs_high, power_high = periodogram(y_train_high_res.high, fs = 1)

    freqs_low, power_low = periodogram(y_train_low_res.low, fs = 1)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(freqs_high, power_high)
    axs[0].set_yscale('log')
    axs[0].set_title('Periodogram High')
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('Power')
    axs[0].grid(True)
    axs[1].plot(freqs_low, power_low)
    axs[1].set_yscale('log')
    axs[1].set_title('Periodogram Low')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Power')
    axs[1].grid(True)
    plt.savefig('images/periodogram.png')
    plt.show()

if __name__ == '__main__':
    periodogram_lags()
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix_lags():
    # set up the data
    volume_lags_query = """
        SELECT volume_1, volume_2, volume_3, volume_4, volume_5, volume_6, volume_7, volume_8, volume_9, volume_10, 
               volume_11, volume_12, volume_13, volume_14, volume_15, volume_16, volume_17, volume_18, volume_19
        FROM X_train
    """

    return_pct_lags_query = """
        SELECT return_pct_lag1, return_pct_lag2, return_pct_lag3, return_pct_lag4, return_pct_lag5, return_pct_lag6, return_pct_lag7, 
               return_pct_lag8, return_pct_lag9, return_pct_lag10, return_pct_lag11, return_pct_lag12, return_pct_lag13, return_pct_lag14, 
               return_pct_lag15, return_pct_lag16, return_pct_lag17, return_pct_lag18, return_pct_lag19
        FROM X_train
    """

    range_lags_query = """
        SELECT range_lag1, range_lag2, range_lag3, range_lag4, range_lag5, range_lag6, range_lag7, range_lag8, range_lag9, range_lag10, 
               range_lag11, range_lag12, range_lag13, range_lag14, range_lag15, range_lag16, range_lag17, range_lag18, range_lag19
        FROM X_train
    """

    high_res_lags_query = """
        SELECT high_res_lag1, high_res_lag2, high_res_lag3, high_res_lag4, high_res_lag5, high_res_lag6, high_res_lag7, 
               high_res_lag8, high_res_lag9, high_res_lag10, high_res_lag11, high_res_lag12, high_res_lag13, high_res_lag14, 
               high_res_lag15, high_res_lag16, high_res_lag17, high_res_lag18, high_res_lag19
        FROM X_train
    """

    low_res_lags_query = """
        SELECT low_res_lag1, low_res_lag2, low_res_lag3, low_res_lag4, low_res_lag5, low_res_lag6, low_res_lag7, 
               low_res_lag8, low_res_lag9, low_res_lag10, low_res_lag11, low_res_lag12, low_res_lag13, low_res_lag14, 
               low_res_lag15, low_res_lag16, low_res_lag17, low_res_lag18, low_res_lag19
        FROM X_train
    """

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        volume_lags = pd.read_sql_query(volume_lags_query, conn)
        return_pct_lags = pd.read_sql_query(return_pct_lags_query, conn)
        range_lags = pd.read_sql_query(range_lags_query, conn)
        high_res_lags = pd.read_sql_query(high_res_lags_query, conn)
        low_res_lags = pd.read_sql_query(low_res_lags_query, conn)

    # create and plot a correlation matrix between the different lags of one feature
    volume_lags.columns = [i for i in range(1, 20)]
    return_pct_lags.columns = [i for i in range(1, 20)]
    range_lags.columns = [i for i in range(1, 20)]
    high_res_lags.columns = [i for i in range(1, 20)]
    low_res_lags.columns = [i for i in range(1, 20)]
    corr_volume_lags = volume_lags.corr()
    corr_return_pct_lags = return_pct_lags.corr()
    corr_range_lags = range_lags.corr()
    corr_high_res_lags = high_res_lags.corr()
    corr_low_res_lags = low_res_lags.corr()
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    ax = ax.flatten()
    sns.heatmap(corr_volume_lags, cmap="YlGnBu", center = 0, ax=ax[0])
    ax[0].set_title("volume_lags")
    sns.heatmap(corr_return_pct_lags, cmap="YlGnBu", center = 0, ax=ax[1])
    ax[1].set_title("return_pct_lags")
    sns.heatmap(corr_range_lags, cmap="YlGnBu", center = 0, ax=ax[2])
    ax[2].set_title("range_lags")
    sns.heatmap(corr_high_res_lags, cmap="YlGnBu", center = 0, ax=ax[4])
    ax[4].set_title("high_res_lags")
    sns.heatmap(corr_low_res_lags, cmap="YlGnBu", center = 0, ax=ax[5])
    ax[5].set_title("low_res_lags")
    fig.delaxes(ax[3])
    fig.suptitle("Correlation Matrices for Lag Features")
    plt.savefig('images/correlation_matrix_lags.png')
    plt.show()

if __name__ == '__main__':
    correlation_matrix_lags()
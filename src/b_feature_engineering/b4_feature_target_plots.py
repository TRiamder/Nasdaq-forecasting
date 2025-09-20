import sqlite3
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


def investigating_features():
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query("SELECT * FROM nasdaq_macros_added", conn)

    df.set_index('date', inplace=True)
    df_train = df.loc[:'2021-12-31']

    # create and plot feature-target relations
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    axs = axs.flatten()
    sns.lineplot(x=df_train.open, y=df_train.high, label='High', ax=axs[0])
    sns.lineplot(x=df_train.open, y=df_train.low, label='Low', ax=axs[0])
    axs[0].legend(loc='upper left')
    axs[0].set_xlabel('Opening Price')
    axs[0].set_ylabel('Price')
    axs[0].set_title('Opening Pice')
    sns.lineplot(x=df_train.CPIAUCSL_30, y=df_train.high, label='High', ax=axs[1])
    sns.lineplot(x=df_train.CPIAUCSL_30, y=df_train.low, label='Low', ax=axs[1])
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel('CPI')
    axs[1].set_ylabel('Price')
    axs[1].set_title('CPI Rate')
    sns.lineplot(x=df_train.UNRATE_30, y=df_train.high, label='High', ax=axs[2])
    sns.lineplot(x=df_train.UNRATE_30, y=df_train.low, label='Low', ax=axs[2])
    axs[2].legend(loc='upper left')
    axs[2].set_xlabel('Unemployment Rate')
    axs[2].set_ylabel('Price')
    axs[2].set_title('Unemployment Rate')
    sns.lineplot(x=df_train.DFF_1, y=df_train.high, label='High', ax=axs[3])
    sns.lineplot(x=df_train.DFF_1, y=df_train.low, label='Low', ax=axs[3])
    axs[3].legend(loc='upper left')
    axs[3].set_xlabel('Federal Funds Rate')
    axs[3].set_ylabel('Price')
    axs[3].set_title('Federal Funds Rate')
    sns.lineplot(x=df_train.DGS10_1, y=df_train.high, label='High', ax=axs[4])
    sns.lineplot(x=df_train.DGS10_1, y=df_train.low, label='Low', ax=axs[4])
    axs[4].legend(loc='upper left')
    axs[4].set_xlabel('10 Year Treasury Rate')
    axs[4].set_ylabel('Price')
    axs[4].set_title('10 Year Treasury Rate')
    sns.lineplot(x=df_train.month_of_year, y=df_train.high, label='High', ax=axs[5])
    sns.lineplot(x=df_train.month_of_year, y=df_train.low, label='Low', ax=axs[5])
    axs[5].legend(loc='upper left')
    axs[5].set_xlabel('Month of the Year')
    axs[5].set_ylabel('Price')
    axs[5].set_title('Month of the Year')
    sns.lineplot(x=df_train.timestep, y=df_train.high, label='High', ax=axs[6])
    sns.lineplot(x=df_train.timestep, y=df_train.low, label='Low', ax=axs[6])
    axs[6].legend(loc='upper left')
    axs[6].set_xlabel('Timestep')
    axs[6].set_ylabel('Price')
    axs[6].set_title('Timestep')
    fig.delaxes(axs[7])
    fig.suptitle("Nasdaq Daily High & Low vs specific features - Training Range")
    plt.savefig('images/feature_target_plots.png')
    plt.show()

if __name__ == '__main__':
    investigating_features()
import sqlite3
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

def investigating():
    # set up the data
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query("SELECT * FROM joined_cleaned", conn)

    df.set_index('date', inplace=True)
    df_train = df.loc[:'2021-12-31']
    df_train.index = pd.to_datetime(df_train.index, format='%Y-%m-%d')

    # create and plot linegraph
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=df_train.index, y=df_train.high, label='High')
    sns.lineplot(x=df_train.index, y=df_train.low, label='Low')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Nasdaq Daily High & Low - Training Range')
    plt.savefig('images/investigating.png')
    plt.show()


if __name__ == '__main__':
    investigating()
import sqlite3
import pandas as pd

def clean_data():
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        joined = pd.read_sql_query("SELECT * FROM nasdaq_macros_joined", conn)

    # filling NA-values in columns that hold macro rates, with the previous rate in time
    cols_to_fill = ['DGS10', 'DFF', 'CPIAUCSL', 'UNRATE']
    joined[cols_to_fill] = joined[cols_to_fill].ffill()
    joined.dropna(inplace=True)

    # shifting feature columns with data that gets published monthly by 30 to prevent lookahead,
    # macro rates for the current month usually get published at the middle of the following month,
    cols_to_shift30 = ['CPIAUCSL', 'UNRATE']
    joined[cols_to_shift30] = joined[cols_to_shift30].shift(30)
    joined.rename(columns={'CPIAUCSL': 'CPIAUCSL_30', 'UNRATE' : 'UNRATE_30'}, inplace=True)

    # shifting feature columns with data that gets published daily by 1 to prevent lookahead
    # final values are available by the next day
    cols_to_shift1 = ['close', 'volume', 'DGS10', 'DFF']
    joined[cols_to_shift1] = joined[cols_to_shift1].shift(1)
    joined.rename(columns={'close': 'close_1', 'volume' : 'volume_1', 'DGS10' : 'DGS10_1', 'DFF' : 'DFF_1'}, inplace=True)

    #print(joined.head())
    #print(joined.info())

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        joined.to_sql('joined_cleaned', conn, if_exists='replace', index=False)

if __name__ == '__main__':
    clean_data()
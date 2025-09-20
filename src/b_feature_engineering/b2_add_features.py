import sqlite3
import pandas as pd

def add_features():
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query("SELECT * FROM joined_cleaned", conn)

    # add time-features
    df['timestep'] = df.index

    # add seasonal features
    df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")
    df["day_of_week"] = df.date.dt.dayofweek
    df["month_of_year"] = df.date.dt.month

    # add lag-features
    df["return_pct_lag1"] = df.close_1.pct_change() * 100

    df["range_lag1"] = df.high - df.low
    df.range_lag1 = df.range_lag1.shift(1)

    lags = list(range(2,20))
    for lag in lags:
        df[f"volume_{lag}"] = df["volume_1"].shift(lag - 1)
        df[f"return_pct_lag{lag}"] = df["return_pct_lag1"].shift(lag - 1)
        df[f"range_lag{lag}"] = df["range_lag1"].shift(lag - 1)

    # drop NA values from the table
    df.dropna(inplace=True)

    #print(df.head())
    #print(df.info())
    #print(df.columns)

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df.to_sql('nasdaq_macros_added', conn, if_exists='replace', index=False)

if __name__ == '__main__':
    add_features()
import sqlite3
import pandas as pd

def avg_daily_ranges():
    # set up data
    query = """
        SELECT `date`, high, low
        FROM nasdaq_macros_added
    """

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query(query, conn)

    df.set_index('date', inplace=True)

    # split data into train, validation and test data
    train_data = df.loc[:'2021-12-31']
    val_data = df.loc['2022-01-01': '2023-12-31']
    test_data = df.loc['2024-01-01':]

    # average daily range for the corresponding data section
    daily_range_train = train_data.high - train_data.low
    avg_daily_range_train = daily_range_train.mean()
    print(f'Average daily range in training set: {avg_daily_range_train}')

    daily_range_val = val_data.high - val_data.low
    avg_daily_range_val = daily_range_val.mean()
    print(f'Average daily range in validation set: {avg_daily_range_val}')

    daily_range_test = test_data.high - test_data.low
    avg_daily_range_test = daily_range_test.mean()
    print(f'Average daily range in test set: {avg_daily_range_test}')

if __name__ == '__main__':
    avg_daily_ranges()
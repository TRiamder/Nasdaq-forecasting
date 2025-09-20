import sqlite3
import pandas as pd

def inspect_data():
    # load the tables out of the nasdaq_macros.db database
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        Nasdaq_Prices = pd.read_sql_query("SELECT * FROM nasdaq_prices", conn)
        Treasury_Yield = pd.read_sql_query("SELECT * FROM treasury_yield_10y", conn)
        Fed_Funds_Rate = pd.read_sql_query("SELECT * FROM fed_funds_rate", conn)
        CPI = pd.read_sql_query("SELECT * FROM cpi", conn)
        Unemployment_Rate = pd.read_sql_query("SELECT * FROM unemployment_rate", conn)

    tables = [Nasdaq_Prices, Treasury_Yield, Fed_Funds_Rate, CPI, Unemployment_Rate]

    # print the tables to inspect them
    for table in tables:
        print(table.head())
        print(table.info())
        print('--' * 40)

if __name__ == '__main__':
    inspect_data()
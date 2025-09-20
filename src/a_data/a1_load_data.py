import sqlite3
import pandas_datareader.data as web
import requests
import time
import pandas as pd

def load_nasdaq():
    API_KEY = 'PRIVATE KEY' # Only available when subscribing to EODHD
    symbol = 'QQQ.US'

    url = (
        f"https://eodhd.com/api/eod/{symbol}?api_token={API_KEY}&from=2009-01-01&to=2025-08-31&fmt=json"
    )

    resp = requests.get(url)
    data = resp.json()

    df = pd.DataFrame(data)
    df.drop(columns='adjusted_close', inplace=True)
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df.to_sql('nasdaq_prices', conn, if_exists='replace', index=False)
    print('Successfully downloaded the nasdaq data.')

def load_macros():
    symbols = {
        'DGS10' : 'treasury_yield_10y',
        'DFF' : 'fed_funds_rate',
        'CPIAUCSL' : 'cpi',
        'UNRATE' : 'unemployment_rate'
    }
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        for key,name in symbols.items():
            macro = web.DataReader(key, 'fred', start='2009-01-01', end='2025-08-31')
            macro.reset_index(inplace=True)
            macro.to_sql(name, conn, if_exists='replace', index=False)

    print('Successfully downloaded the macros data.')

if __name__ == '__main__':
    load_nasdaq()
    load_macros()

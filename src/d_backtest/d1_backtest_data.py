import sqlite3
import pandas as pd
import requests
import time

def load_backtest_data():
    # load intraday data for the backtest of trading model
    api_token = 'YOUR API TOKEN' # Only available when subscribing to EODHD
    symbol = 'QQQ.US'
    interval = '1h'
    start = int(time.mktime(time.strptime('2022-01-01', '%Y-%m-%d')))
    end = int(time.mktime(time.strptime('2025-08-31', '%Y-%m-%d')))

    url = (
        f"https://eodhd.com/api/intraday/{symbol}?api_token={api_token}&interval={interval}&from={start}&to={end}&fmt=json"
    )

    resp = requests.get(url)
    data = resp.json()

    df = pd.DataFrame(data)
    #print(df)
    #print(df.info())

    with sqlite3.connect('raw_data/nasdaq_backtest_intraday.db') as conn:
        df.to_sql('nasdaq_prices_1h', conn, if_exists='replace')

if __name__ == '__main__':
    load_backtest_data()
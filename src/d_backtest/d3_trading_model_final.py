import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def trading_model_final():
    # set up the data
    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        data = pd.read_sql_query("SELECT * FROM data_trading_model_val", conn)
    data.set_index('date', inplace=True)

    # performance of the final trading model during the validation period
    equity = [1000.0]
    capital = 1000.0
    risk_per_trade = 1.0

    for date, candle_d in data.iterrows():
        pred_high_d = candle_d['pred_high']
        pred_low_d = candle_d['pred_low']
        close_d = candle_d['close']

        pnl = 0
        entry_point = 0.8767071031532336
        sl_point = 4.9629424445576005
        tp_point = 3.0149741031683983
        entry = pred_low_d + entry_point
        sl = pred_low_d - sl_point
        tp = pred_high_d - tp_point
        min_range = 3.0047352533852054
        trade_placed = False
        tp_hit_first = False
        sl_hit_first = False

        if pred_high_d - pred_low_d > min_range:
            query_intraday = f"""
                    SELECT `datetime`, `open`, high, low, `close`
                    FROM nasdaq_prices_1h
                    WHERE DATETIME(DATE(datetime)) = '{date}'
                """

            with sqlite3.connect('raw_data/nasdaq_backtest_intraday.db') as conn:
                intraday = pd.read_sql_query(query_intraday, conn)

            for datetime, candle_h in intraday.iterrows():
                high_h = candle_h['high']
                low_h = candle_h['low']

                if low_h < entry and not tp_hit_first:
                    trade_placed = True
                if low_h < sl and not tp_hit_first:
                    sl_hit_first = True
                if high_h > tp and not sl_hit_first:
                    tp_hit_first = True
                if high_h > tp and low_h < sl:
                    sl_hit_first = True

            if trade_placed:
                if sl_hit_first:
                    pnl = (sl - entry) * risk_per_trade
                elif tp_hit_first:
                    pnl = (tp - entry) * risk_per_trade
                else:
                    pnl = (close_d - entry) * risk_per_trade

        capital += pnl
        equity.append(capital)

    # calculate the sharpe ratio of the model
    data["equity"] = equity[1:]
    equity_sharp = pd.Series(equity)
    daily_returns = equity_sharp.pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)

    # visualize the performance of the trading model
    date_parsed = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(date_parsed, data.equity)
    ax.set_title("Equity Curve - Trading Model Backtest - Validation Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    end_equity = equity[-1]
    equity_str = f"End Equity: {end_equity:.2f}"
    ax.text(
        0.99, 0.5, equity_str, transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )
    sharpe_ratio_str = f"Sharpe Ratio: {sharpe_ratio:.2f}"
    ax.text(
        0.99, 0.45, sharpe_ratio_str, transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )
    plt.savefig('images/equity_curve_val.png')
    plt.show()
    #print(f"sharpe ratio: {sharpe_ratio}")
    #print(f"average size of winning trades: {equity_spread.abs().mean()}")

if __name__ == "__main__":
    trading_model_final()
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def trading_model_test():
    # set up the data
    query1 = """
        SELECT `date`, `open`, high, low
        FROM nasdaq_macros_added
        WHERE DATE(`Date`) > '2023-12-31';
    """

    query2 = """
        SELECT DATETIME(`date`) AS `date`, `close`
        FROM nasdaq_macros_joined
        WHERE DATE(`Date`) > '2023-12-31'
    """

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df_test = pd.read_sql_query(query1, conn)
        close = pd.read_sql_query(query2, conn)
        y_test_pred_lin_reg = pd.read_sql_query("SELECT * FROM test_predictions", conn)
        pred_high_test_xgb = pd.read_sql_query("SELECT * FROM pred_high_test_xgb", conn)
        pred_low_test_xgb = pd.read_sql_query("SELECT * FROM pred_low_test_xgb", conn)

    tables= [df_test, close, y_test_pred_lin_reg, pred_high_test_xgb, pred_low_test_xgb]

    for table in tables:
        table.set_index('date', inplace=True)

    y_test_pred_lin_reg.columns = ['pred_high', 'pred_low']
    close.dropna(inplace=True)

    pred_high_sum = y_test_pred_lin_reg.pred_high + pred_high_test_xgb.pred_high
    pred_low_sum = y_test_pred_lin_reg.pred_low + pred_low_test_xgb.pred_low

    data = pd.concat([df_test, close, pred_high_sum, pred_low_sum], axis=1)

    # performance of the trading model during the test period
    equity = [1000.0]
    capital = 1000.0
    risk_per_trade = 1.0

    for date, candle_d in data.iterrows():
        pred_high_d = candle_d['pred_high']
        pred_low_d = candle_d['pred_low']
        close_d = candle_d['close']

        pnl = 0.0
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
    data["equity"] = equity[1:]

    # calculate the sharpe ratio of the model
    equity_sharp = pd.Series(equity)
    daily_returns = equity_sharp.pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)

    # visualize the performance of the trading model
    date_parsed = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(date_parsed, data.equity)
    ax.set_title("Equity Curve - Trading Model Backtest - Test Period (unseen data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")

    # calculate the return
    end_equity = equity[-1]
    end_return = (end_equity / 1000.0 - 1) * 100
    return_str = f"Return: {end_return:.2f}%"

    ax.text(
        0.99, 0.4, return_str, transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )
    sharpe_ratio_str = f"Sharpe Ratio: {sharpe_ratio:.2f}"
    ax.text(
        0.99, 0.35, sharpe_ratio_str, transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.savefig('results/plots/equity_curve_test.png')
    plt.show()

    # include the transaction costs in the equity curve
    equity_for_costs = pd.Series(equity)
    equity_diffs = equity_for_costs.diff().dropna()
    equity_costs = [1000.0]
    for i in equity_diffs:
        if i == 0:
            equity_costs.append(equity_costs[-1])
        else:
            equity_costs.append((i - 0.1) + equity_costs[-1])
    equity_costs_series = pd.Series(equity_costs[1:])

    # Nasdaq buy and hold return for comparison
    nasdaq = close.close
    nasdaq_return = nasdaq + 1000.0 - nasdaq.iloc[0]

    # visualize the performance of the trading model including transaction costs
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(date_parsed, equity_costs_series, label='Equity Curve Backtesting the Trading Model')
    ax.plot(date_parsed, nasdaq_return, label='Nasdaq Buy & Hold Return')
    ax.legend(loc='upper left')
    ax.set_title("Equity Curve - Trading Model Backtest - Test Period (unseen data) - Transaction Costs Included")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")

    # calculate the return
    end_equity_costs = equity_costs[-1]
    end_return_costs = (end_equity_costs / 1000.0 - 1) * 100
    return_costs_str = f"Model Return: {end_return_costs:.2f}%"

    ax.text(
        0.99, 0.40, return_costs_str, transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    # calculate the sharpe ratio
    equity_sharp_costs = pd.Series(equity_costs)
    daily_returns_costs = equity_sharp_costs.pct_change().dropna()
    sharpe_ratio_costs = daily_returns_costs.mean() / daily_returns_costs.std(ddof=1) * np.sqrt(252)
    sharpe_ratio_costs_str = f"Model Sharpe Ratio: {sharpe_ratio_costs:.2f}"

    ax.text(
        0.99, 0.35, sharpe_ratio_costs_str, transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.savefig('results/plots/equity_curve_costs_test.png')
    plt.show()

if __name__ == "__main__":
    trading_model_test()
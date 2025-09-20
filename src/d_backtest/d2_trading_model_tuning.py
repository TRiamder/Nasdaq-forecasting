import sqlite3
import pandas as pd
import optuna
import numpy as np

def trading_model_tuning():
    # set up the data
    query1 = """
        SELECT `date`, `open`, high, low
        FROM nasdaq_macros_added
        WHERE '2021-12-31' < DATE(`date`) AND DATE(`date`) < '2024-01-01'
    """

    query2 = """
        SELECT DATETIME(`date`) AS `date`, `close`
        FROM nasdaq_macros_joined
        WHERE '2021-12-31' < DATE(`date`) AND DATE(`date`) < '2024-01-01'
    """

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df_val = pd.read_sql_query(query1, conn)
        close = pd.read_sql_query(query2, conn)
        y_val_pred_lin_reg = pd.read_sql_query("SELECT * FROM val_predictions", conn)
        pred_high_val_xgb = pd.read_sql_query("SELECT * FROM pred_high_val_xgb", conn)
        pred_low_val_xgb = pd.read_sql_query("SELECT * FROM pred_low_val_xgb", conn)

    tables= [df_val, close, y_val_pred_lin_reg, pred_high_val_xgb, pred_low_val_xgb]

    for table in tables:
        table.set_index('date', inplace=True)

    y_val_pred_lin_reg.columns = ['pred_high', 'pred_low']
    close.dropna(inplace=True)

    pred_high_sum = y_val_pred_lin_reg.pred_high + pred_high_val_xgb.pred_high
    pred_low_sum = y_val_pred_lin_reg.pred_low + pred_low_val_xgb.pred_low

    data = pd.concat([df_val, close, pred_high_sum, pred_low_sum], axis=1)

    # create trading model and tune it on the backtesting results of the validation data
    def objective(trial):
        equity_study = [1000.0]
        capital_study = 1000.0
        risk_per_trade_study = 1.0

        for date, candle_d in data.iterrows():
            pred_high_d = candle_d['pred_high']
            pred_low_d = candle_d['pred_low']
            close_d = candle_d['close']

            pnl = 0.0
            entry_point = trial.suggest_float("entry", 0.0, 3.0)
            sl_point = trial.suggest_float("sl", 0.0, 5.0)
            tp_point = trial.suggest_float("tp", 0.0, 5.0)
            entry = pred_low_d + entry_point
            sl = pred_low_d - sl_point
            tp = pred_high_d - tp_point
            min_range = trial.suggest_float("min_range", 3, 9)
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

                    if low_h < entry and high_h > entry and not tp_hit_first:
                        trade_placed = True
                    if low_h < sl and not tp_hit_first:
                        sl_hit_first = True
                    if high_h > tp and not sl_hit_first:
                        tp_hit_first = True
                    if high_h > tp and low_h < sl:
                        sl_hit_first = True

                if trade_placed:
                    if sl_hit_first:
                        pnl = (sl - entry) * risk_per_trade_study
                    elif tp_hit_first:
                        pnl = (tp - entry) * risk_per_trade_study
                    else:
                        pnl = (close_d - entry) * risk_per_trade_study

            capital_study += pnl
            equity_study.append(capital_study)
        equity = pd.Series(equity_study)
        daily_returns = equity.pct_change().dropna()
        if daily_returns.std(ddof=1) == 0.0:
            return 0
        sharpe_ratio = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
        return capital_study * sharpe_ratio


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print('best result with the following parameters:')
    print(study.best_params)

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        data.to_sql('data_trading_model_val', conn, if_exists="replace")

if __name__ == '__main__':
    trading_model_tuning()
import sqlite3
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import optuna

def xgboost_tuning():
    # set up the data
    query_train = """
        SELECT `date`, DFF_1, timestep,
            `open`, CPIAUCSL_30, UNRATE_30, DGS10_1, 
            volume_1,
            return_pct_lag1, return_pct_lag2, return_pct_lag9,
            range_lag1, range_lag2, range_lag3, range_lag4, range_lag5, range_lag6, range_lag8, range_lag9,
            high_res_lag1, high_res_lag2, high_res_lag3, high_res_lag4, high_res_lag5, high_res_lag6, high_res_lag7,
            low_res_lag1, low_res_lag2, low_res_lag3, low_res_lag4
        FROM X_train
    """

    query_val = """
        SELECT `date`, DFF_1, timestep,
            `open`, CPIAUCSL_30, UNRATE_30, DGS10_1, 
            volume_1,
            return_pct_lag1, return_pct_lag2, return_pct_lag9,
            range_lag1, range_lag2, range_lag3, range_lag4, range_lag5, range_lag6, range_lag8, range_lag9,
            high_res_lag1, high_res_lag2, high_res_lag3, high_res_lag4, high_res_lag5, high_res_lag6, high_res_lag7,
            low_res_lag1, low_res_lag2, low_res_lag3, low_res_lag4
        FROM X_val
    """

    query_test = """
        SELECT `date`, DFF_1, timestep,
            `open`, CPIAUCSL_30, UNRATE_30, DGS10_1,
            volume_1,
            return_pct_lag1, return_pct_lag2, return_pct_lag9,
            range_lag1, range_lag2, range_lag3, range_lag4, range_lag5, range_lag6, range_lag8, range_lag9,
            high_res_lag1, high_res_lag2, high_res_lag3, high_res_lag4, high_res_lag5, high_res_lag6, high_res_lag7,
            low_res_lag1, low_res_lag2, low_res_lag3, low_res_lag4
        FROM X_test
    """

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        X_train = pd.read_sql_query(query_train, conn)
        X_val = pd.read_sql_query(query_val, conn)
        X_test = pd.read_sql_query(query_test, conn)
        y_train_high_res = pd.read_sql_query('SELECT * FROM y_train_high_res', conn)
        y_train_low_res = pd.read_sql_query('SELECT * FROM y_train_low_res', conn)
        y_val_high_res = pd.read_sql_query('SELECT * FROM y_val_high_res', conn)
        y_val_low_res = pd.read_sql_query('SELECT * FROM y_val_low_res', conn)
        y_test_high_res = pd.read_sql_query('SELECT * FROM y_test_high_res', conn)
        y_test_low_res = pd.read_sql_query('SELECT * FROM y_test_low_res', conn)

    tables= [X_train, X_val, X_test, y_train_high_res, y_train_low_res, y_val_high_res, y_val_low_res, y_test_high_res, y_test_low_res]

    for table in tables:
        table.set_index('date', inplace=True)

    # tune XGB model on validation data
    def xgboost(xgb_model_high, xgb_model_low):
        xgb_model_high.fit(X_train, y_train_high_res, eval_set=[(X_train, y_train_high_res)], verbose=False)
        xgb_model_low.fit(X_train, y_train_low_res, eval_set = [(X_train, y_train_low_res)], verbose=False)
        pred_high_val = pd.Series(xgb_model_high.predict(X_val), index=X_val.index)
        pred_low_val = pd.Series(xgb_model_low.predict(X_val), index=X_val.index)
        return pred_high_val, pred_low_val

    def objective(trial):
        xgb_params_high = dict(
            n_estimators=trial.suggest_int("n_estimators_high", 400, 2000),
            early_stopping_rounds=trial.suggest_int("early_stopping_rounds_high", 5, 75),
            max_depth=trial.suggest_int("max_depth_high", 2, 10),
            learning_rate=trial.suggest_float("learning_rate_high", 1e-5, 3e-1, log=True),
            min_child_weight=trial.suggest_int("min_child_weight_high", 1, 10),
            gamma=trial.suggest_float("gamma_high", 0, 5),
            colsample_bytree=trial.suggest_float("colsample_bytree_high", 0.2, 1),
            subsample=trial.suggest_float("subsample_high", 0.2, 1),
            reg_alpha=trial.suggest_float("reg_alpha_high", 1e-8, 1e2, log=True),
            reg_lambda=trial.suggest_float("reg_lambda_high", 1e-8, 10.0, log=True),
        )
        xgb_params_low = dict(
            n_estimators=trial.suggest_int("n_estimators_low", 400, 2000),
            early_stopping_rounds=trial.suggest_int("early_stopping_rounds_low", 5, 75),
            max_depth=trial.suggest_int("max_depth_low", 2, 10),
            learning_rate=trial.suggest_float("learning_rate_low", 1e-5, 3e-1, log=True),
            min_child_weight=trial.suggest_int("min_child_weight_low", 1, 10),
            gamma=trial.suggest_float("gamma_low", 0, 5),
            colsample_bytree=trial.suggest_float("colsample_bytree_low", 0.2, 1),
            subsample=trial.suggest_float("subsample_low", 0.2, 1),
            reg_alpha=trial.suggest_float("reg_alpha_low", 1e-8, 1e2, log=True),
            reg_lambda=trial.suggest_float("reg_lambda_low", 1e-8, 10.0, log=True),
        )
        pred_high_val, pred_low_val = xgboost(XGBRegressor(**xgb_params_high), XGBRegressor(**xgb_params_low))
        mae_val_high = mean_absolute_error(y_val_high_res, pred_high_val)
        mae_val_low = mean_absolute_error(y_val_low_res, pred_low_val)
        mae_val_avg = (mae_val_high + mae_val_low) / 2
        return mae_val_avg

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print('best result with the following parameters:')
    print(study.best_params)

if __name__ == '__main__':
    xgboost_tuning()
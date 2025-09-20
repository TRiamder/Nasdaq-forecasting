import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.metrics import mean_absolute_error

def xgboost_final():
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

    with sqlite3.connect("raw_data/nasdaq_macros.db") as conn:
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


    # train XGB model
    final_xgb_params_high = dict(
        n_estimators=1676,
        early_stopping_rounds=36,
        max_depth=7,
        learning_rate=0.0004841764103594442,
        min_child_weight=4,
        gamma=2.7113444054658227,
        colsample_bytree=0.9298638403972058,
        subsample=0.3128153336927164,
        reg_alpha=0.00010515936803605924,
        reg_lambda=0.0008057912466261512
    )

    final_xgb_params_low = dict(
        n_estimators=465,
        early_stopping_rounds=13,
        max_depth=7,
        learning_rate=0.0020863408423926356,
        min_child_weight=2,
        gamma=1.1472321241507282,
        colsample_bytree=0.32050449281388,
        subsample=0.43337713381945364,
        reg_alpha=2.714050593397846,
        reg_lambda=7.334719917567699
    )

    final_xgb_model_high = XGBRegressor(**final_xgb_params_high).fit(X_train, y_train_high_res, eval_set=[(X_train, y_train_high_res)], verbose=False)
    final_xgb_model_low = XGBRegressor(**final_xgb_params_low).fit(X_train, y_train_low_res, eval_set = [(X_train, y_train_low_res)], verbose=False)

    # plot error distribution of the model on validation data
    pred_high_val = pd.Series(final_xgb_model_high.predict(X_val), index=X_val.index, name="pred_high")
    pred_low_val = pd.Series(final_xgb_model_low.predict(X_val), index=X_val.index, name="pred_low")
    error_high = y_val_high_res.high - pred_high_val
    error_low = y_val_low_res.low - pred_low_val

    plt.figure(figsize=(12, 8))
    sns.kdeplot(data= error_high, label="Error High = High Residual - Predicted High Residual")
    sns.kdeplot(data= error_low, label="Error Low = Low Residual - Predicted Low Residual")
    plt.legend()
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('Error of the XGB model in the validation range')
    plt.savefig('images/error_dis_xgb_model.png')
    plt.show()

    # MAE in validation range
    mae_val_high = mean_absolute_error(y_val_high_res, pred_high_val)
    mae_val_low = mean_absolute_error(y_val_low_res, pred_low_val)
    mae_val_avg = (mae_val_high + mae_val_low) / 2
    print(f"MAE in validation range: {mae_val_avg}")

    # test the model
    # MAE in test range
    pred_high_test = pd.Series(final_xgb_model_high.predict(X_test), index=X_test.index, name="pred_high")
    pred_low_test = pd.Series(final_xgb_model_low.predict(X_test), index=X_test.index, name="pred_low")
    mae_test_high = mean_absolute_error(y_test_high_res, pred_high_test)
    mae_test_low = mean_absolute_error(y_test_low_res, pred_low_test)
    mae_test_avg = (mae_test_high + mae_test_low) / 2
    print(f"MAE in test range: {mae_test_avg}")

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        pred_high_val.to_sql('pred_high_val_xgb', conn, if_exists='replace')
        pred_low_val.to_sql('pred_low_val_xgb', conn, if_exists='replace')
        pred_high_test.to_sql('pred_high_test_xgb', conn, if_exists='replace')
        pred_low_test.to_sql('pred_low_test_xgb', conn, if_exists='replace')

if __name__ == '__main__':
    xgboost_final()
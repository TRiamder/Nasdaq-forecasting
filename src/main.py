from src.a_data.a2_inspect_data import inspect_data
from src.a_data.a3_join_data import join_data
from src.a_data.a4_clean_data import clean_data
from src.b_feature_engineering.b1_investigating import investigating
from src.b_feature_engineering.b2_add_features import add_features
from src.b_feature_engineering.b3_mutual_information import mutual_information_pre
from src.b_feature_engineering.b5_correlation_matrix import correlation_matrix
from src.c_train_model.avg_daily_ranges import avg_daily_ranges
from src.c_train_model.c1_linear_regression import linear_regression
from src.c_train_model.c2_residuals import residuals
from src.c_train_model.c3_mutual_information_res import mutual_information_res
from src.c_train_model.c3_correlation_matrix_lags import correlation_matrix_lags
from src.c_train_model.c3_pacf_lags import pacf_lags
from src.c_train_model.c3_periodogram import periodogram_lags
from src.c_train_model.c5_xgboost_final import xgboost_final
from src.d_backtest.d3_trading_model_final import trading_model_final
from src.d_backtest.d4_trading_model_test import trading_model_test

print("--->")
print("Inspect the raw data")
inspect_data()
input("Press enter to continue...")
print("--->")
print("Join the raw data into one column")
join_data()
input("Press enter to continue...")
print("--->")
print("Clean the data")
clean_data()
input("Press enter to continue...")
print("--->")
print("Investigate the data by visualization")
investigating()
input("Press enter to continue...")
print("--->")
print("Add features that could be useful")
add_features()
input("Press enter to continue...")
print("--->")
print("Mutual information scores between the features and the targets:")
mutual_information_pre()
input("Press enter to continue...")
print("--->")
print("Correlation matrix between the features:")
correlation_matrix()
input("Press enter to continue...")
print("--->")
avg_daily_ranges()
input("Press enter to continue...")
print("--->")
print("Train and test the linear regression model")
linear_regression()
input("Press enter to continue...")
print("--->")
print("Create the target residuals")
residuals()
input("Press enter to continue...")
print("--->")
print("Mutual information scores between the features and the target residuals:")
mutual_information_res()
input("Press enter to continue...")
print("--->")
print("Correlation matrix between the lags of a kind:")
correlation_matrix_lags()
input("Press enter to continue...")
print("--->")
print("PACF plots of the lags:")
pacf_lags()
input("Press enter to continue...")
print("--->")
print("Periodogram of the target residuals:")
periodogram_lags()
input("Press enter to continue...")
print("--->")
print("Train and test the XGB model with the tuned hyperparameters")
xgboost_final()
input("Press enter to continue...")
print("--->")
print("Performance of the trading model with tuned parameters on the validation range:")
trading_model_final()
input("Press enter to continue...")
print("--->")
print("Performance of the trading model with tuned parameters on the test range:")
trading_model_test()
print("Done!!!")
print("Thank you for your time!")

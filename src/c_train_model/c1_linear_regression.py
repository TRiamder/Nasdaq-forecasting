import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
pd.plotting.register_matplotlib_converters()
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def linear_regression():
    # set up the data
    query1 = """
        SELECT `date`, `open`, high, low
        FROM nasdaq_macros_added
    """

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query(query1, conn)

    df.set_index('date', inplace=True)

    train_data = df.loc[:'2021-12-31']
    val_data = df.loc['2022-01-01':'2023-12-31']
    test_data = df.loc['2024-01-01':]

    targets = ["high", "low"]

    y_train = train_data[targets]
    y_val = val_data[targets]
    y_test = test_data[targets]

    X_train = train_data.drop(targets, axis=1)
    X_val = val_data.drop(targets, axis=1)
    X_test = test_data.drop(targets, axis=1)

    # scale feature data
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # train and predict with linear regression model
    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train)
    y_train_pred = pd.DataFrame(lin_model.predict(X_train_scaled), columns = y_train.columns, index = X_train_scaled.index)
    y_val_pred = pd.DataFrame(lin_model.predict(X_val_scaled), columns = y_val.columns, index = X_val_scaled.index)
    y_test_pred = pd.DataFrame(lin_model.predict(X_test_scaled), columns = y_test.columns, index = X_test_scaled.index)

    # mean absolute error scores of the model on the data splits
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"MAE train: {mae_train}")
    print(f"MAE val: {mae_val}")
    print(f"MAE test: {mae_test}")

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        y_train_pred.to_sql('train_predictions', conn, if_exists='replace')
        y_val_pred.to_sql('val_predictions', conn, if_exists='replace')
        y_test_pred.to_sql('test_predictions', conn, if_exists='replace')

if __name__ == '__main__':
    linear_regression()
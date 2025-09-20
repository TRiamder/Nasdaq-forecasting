import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix():
    # set up data
    query = """
        SELECT `date`, `open`, timestep, CPIAUCSL_30, month_of_year, high, low
        FROM nasdaq_macros_added
    """

    with sqlite3.connect('raw_data/nasdaq_macros.db') as conn:
        df = pd.read_sql_query(query, conn)

    df.set_index('date', inplace=True)

    # split train and test data
    train_data = df.loc[:'2021-12-31']

    targets = ["high", "low"]

    X_train = train_data.drop(targets, axis=1)

    # Transform features based on the graph from the feature-target plots
    X_train['CPIAUCSL_30_sq'] = X_train.CPIAUCSL_30 ** 2
    X_train.drop('CPIAUCSL_30', axis=1, inplace=True)
    X_train['timestep_sq'] = X_train.timestep ** 2
    X_train.drop('timestep', axis=1, inplace=True)

    # create correlation matrix
    corr = X_train.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="YlGnBu", center = 0)
    plt.title("Correlation Matrix - Linear Features")
    plt.savefig('images/correlation_matrix.png')
    plt.show()

if __name__ == '__main__':
    correlation_matrix()
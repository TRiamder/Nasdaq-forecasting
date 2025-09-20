from sklearn.feature_selection import mutual_info_regression
import pandas as pd

# function to calculate the mutual information scores
def mutual_information(X, y_high, y_low):
    mi_high = mutual_info_regression(X, y_high)
    mi_low = mutual_info_regression(X, y_low)
    mi_high = pd.Series(mi_high, name="MI High", index=X.columns)
    mi_low = pd.Series(mi_low, name="MI Low", index=X.columns)
    mi_high = mi_high.sort_values(ascending=False)
    mi_low = mi_low.sort_values(ascending=False)
    return mi_high, mi_low
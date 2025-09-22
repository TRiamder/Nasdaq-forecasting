# Technical Insights - Nasdaq ML Trading Project

This document provides a deeper technical insight into the project.

## Data
**1. Loading Data**

- **Market Data** : QQQ OHLCV daily data (downloaded via EODHD)
- **Macroeconomic Data** : 10y Treasury Rate, Federal Funds Rate, CPI Rate and 
Unemployment Rate (downloaded via fred)
- **Reasoning** : The goal was to combine asset-specific information (price action) 
with broader economic indicators that may influence intraday behavior.

**2. Data Preprocessing**

- **Joining** : Joining the data from the different sources by the date.
- **Cleaning** : Forward-filling missing values for macro data (monthly data).
Shifting data to create real-time accessibility (avoid lookahead). Dropping NA values created by joining and shifting.
- **Insight** : It was crucial to use a union join, using a left join on the market data would have caused 
macroeconomic rates released on weekends and bank holidays to be dropped leading to a loss of important data.

**3. Data Splits**

- **Training Range (2009-01-01 to 2021-12-31)** : Train the Machine Learning Models
- **Validation Range (2022-01-01 to 2023-12-31)** : Validate and tune the 
models
- **Testing Range (2024-01-01 to 2025-08-31)** : Test the models

## Feature Engineering
**1. Adding Features**

- **Time Features** : First- and second-order timesteps, Day of week, Month of year
- **Lag Features** : Volume, Daily return (%), Daily range
- **Reasoning** : Second-order timesteps since by looking at the high and low plot 
one can observe a quadratic trend over the training range. 
Add a variety of features that could hold predictive power to investigate them further later.

**2. Feature Selection for Linear Regression**

- **Mutual Information** : Calculating the mutual information scores between 
the features and the targets (daily high and low).
- **Feature-Target Plots** : Plot the features with the highest mutual information score against the targets.
- **Correlation Matrix** : Examine the correlation between the features.
- **Reasoning** : Pipeline to end up with a selected set of features to train the linear regression model on.
Take the features with the highest mutual information score. Visualize their relationship to the target.
Keep the ones that can be described by a linear model and transform them if needed by squaring or cubing for example.
If there is collinearity between some of the final features, only stick to the ones with the highest mutual information score.
Since the linear regression model can not comprehend interactions between the features, it is not profiting to check beyond the mutual information scores.

## ML Modelling Approach
**Boosted Hybrid Setup**

- **Linear Regression** : Capture base trends
- **XGBoost** : Capture interactions
- **Boosted** : Use target residuals as the target variable for the XGBoost model (the difference between the real target values 
and the ones predicted by the linear regression model)

## ML Model
**1. Linear Regression**

- **Feature** : Opening price of the day
- **Targets** : Daily high and low
- **Test Model** : Training Range: MAE: 0.66, Average daily range: 1.81; 
Validation Range: MAE: 2.0, Average daily range: 6.23; Testing Range: MAE: 
2.47; Average daily range: 6.87
- **Insight** : The feature selection pipeline gave me the opening price of the day and the month of the year as features,
but by trying out the different possible combinations by their performance on the training and validation range, 
it turned out that using just the opening price of the day scores best.
Also, it might seem like the model is overfitting by the validation MAE being three times the training MAE,
but also notice that the average daily range more than tripled from the training to the validation range.

**2. Feature Selection for XGBoost** 

- **Create Target Residuals** : Subtract the target predictions of the linear regression 
model from the real target values. Add lag features of the target residuals.
- **PACF Plot** : Examine the partial correlation between the multiple lags of one feature and the original feature (only retain relevant lags).
- **Mutual Information** : Mutual Information scores between every feature and both the targets to find out which features hold the most information about the targets.
- **Correlation Matrix** : Examine the correlation within the different lag features of a kind to avoid collinearity.
- **Periodogram** : Check the target residuals for seasonality.
- **Reasoning** : Determine which lags of one original feature are most relevant by selecting those whose partial autocorrelation lies outside the 95% confidence interval 
(i.e., significantly different from zero under the null hypothesis of white noise). 
Also include lags that are not determined by the PACF plots but that have a higher mutual information score than the other lags in the same plot. 
If any of the resulting features are highly correlated, keep the one with the highest mutual information scores.
Use the periodogram to investigate the target residuals on seasonality.

**3. XGBoost**

- **Features** : Macros, Timesteps, Opening price of the day, Multiple lags 
(see [code](../src/c_train_model/c5_xgboost_final.py) for more detail)
- **Targets** : Daily high and low residuals
- **Hyperparameter Tuning** : Used Optuna to tune the model on the 
validation range.
- **Insights** : Used two models to make predictions for the high and the low 
residual. Did not use seasonal features since the Periodogram showed no 
signs of seasonality, and training the model with them included did not show an increase in the model's performance.
Ended up with my final set of features by further testing multiple combinations around the methodically investigated features.

**4. Test the Combined Hybrid Model (= XGBoost Model Results)**

- **Validation Range** : MAE: 1.94
- **Test Range** : MAE: 2.41
- **Insight** : Note that the improvement of the hybrid model over the linear regression baseline is 
relatively small. The linear regression model captures most of the signal while the XGBoost model 
provides a slight additional benefit. Also keep in mind though that the smaller the MAE gets, further
improvements become increasingly difficult. I also tried training the linear regression model on weaker predictive features 
by which you can shift the model in a way that the XGBoost model is doing most of the heavy lifting, 
but the end results were not as good as this approach.

## Trading Model

**1. Load Data**

- **Load Intraday Data** : QQQ OHLCV hourly data (downloaded via EODHD)
- **Reasoning** : Intraday data to run the backtesting model.

**2. Trading Model**

- **Build Trading Model** : Long-only model that acts on a daily basis and utilizes the QQQ daily high 
prediction, low prediction, closing price and QQQ hourly data. The model buys at a specific price level 
that is determined once the opening price of the day is known, holds the trade until the take 
profit or stop loss gets hit and realizes the profit or loss. Otherwise, if neither gets hit, it sells the position just before the close of the day. 
The model has a slightly pessimistic bias since it treats cases where the hourly candle hits the take profit and 
stop loss at once as a loss. If the take profit was already hit before reaching the entry price, no trade is taken.
- **Tune Trading Model** : Use Optuna to tune the model on the return and the sharpe ratio in validation range. 
Find the best performing values for the entry and the stop loss relative to the predicted low, 
the take profit relative to the predicted high and the minimum predicted high-low range required to take a trade.
- **Reasoning** : By understanding the error distribution of the prediction model it is possible to exploit ranges 
within the predicted daily high–low spreads. The Optuna study is used to find the optimal parameters for the trading strategy 
to best take advantage of this error distribution. I chose a long-only approach to simplify the proof-of-concept, 
focusing on demonstrating the practical usability of machine learning models in trading.  
Additionally, long positions statistically outperform short positions in equity markets.
- **Example** : If it is highly unlikely that the true daily high falls more than 3 points below the predicted high  
and equally unlikely that the true daily low rises more than 3 points above the predicted low
then any predicted high–low range exceeding 6 points creates a tradable opportunity.
In such cases, the model’s error distribution suggests that the inner part of the predicted range  
is very likely to be filled during the trading day.

**3. Test Trading Model**

- **Performance** : Validation Range: Return: 24.94%, Sharpe Ratio: 6.99
**Test Range: Return: 12.25% in 2 years of backtest, Sharpe Ratio: 2.60**
- **Transaction Costs** : After including an approximation of the trading costs: 
Return: 11.12% in 2 years of backtest, Sharpe Ratio: 2.37
- **Reasoning** : Transaction costs were modeled as the combination of slippage, spread, and commissions. 
To ensure a conservative estimation, I subtracted 0.1 points from every simulated trade outcome. 
This pessimistic assumption likely overestimates real-world costs for trading QQQ, 
which is one of the most liquid ETFs with typical round-trip costs closer to 0.03–0.05 points.
- **Caveats & Considerations** : The backtesting results with transaction costs included are only a rough approximation 
and may deviate from real-world performance. By comparing the trading model's performance on the validation and test sets, it appears that the model is overfitted on the validation data.  
Nonetheless, it still demonstrates strong performance on the test set, indicating reasonable out-of-sample robustness.
Even though, the model did not yield a return as good as the Nasdaq, it ensured way less volatility with a still solid return.
Also, the high Sharpe Ratio suggests broad opportunities for applying leverage to boost returns, though this equally increases risk. 
Since the trading model’s parameters are tuned on the error distribution of the machine learning model in a specific range, 
a rolling window approach is likely required to adapt the tuning process to changing market conditions and evolving error distributions.
Furthermore, the model can be extended beyond the long-only approach by also implementing short-selling opportunities.
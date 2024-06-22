# Gold_price_forecasting

## Gold Price Forecasting 
This project investigates the use of machine learning models to forecast gold prices based on historical data.

**Objective:**

* Develop accurate models to predict the closing price of gold for the following day.

**Approach:**

1. **Data Collection and Preprocessing:**
   * Gathered historical gold price data from [[source of data]](https://finance.yahoo.com/quote/GC=F/).
   * Cleaned and preprocessed the data to address missing values, outliers, and ensure consistency.
   * Explored the data to understand patterns and trends using techniques like visualization and statistical analysis.

2. **Model Development and Evaluation:**
   * Implemented and evaluated various machine learning models commonly used for time series forecasting, including:
     * Bayesian Neural Networks (BNNs)
     * Hidden Markov Models (HMMs)
     * Convolutional Neural Networks (CNNs) combined with Long Short-Term Memory (LSTM) networks (CNN-LSTMs)
     * Autoregressive Integrated Moving Average (ARIMA) models (if applicable)
     * Gated Recurrent Units (GRUs)
     * LSTMs
   * Used a rolling window of 60 days to train models on historical data and predict the closing price for the next day.
   * Performed hyperparameter tuning to optimize model performance.
   * Evaluated model accuracy using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
   * Compared the performance of different models to select the best performing one.

**Technologies:**

* Programming Language: Python
* Data Analysis Libraries: Pandas, NumPy
* Machine Learning Libraries: Scikit-learn (if applicable for models like ARIMA), Keras (for deep learning models like CNN-LSTM)
* Data Visualization: Matplotlib

# EUR/USD Exchange Rate Prediction

This project explores the use of machine learning and deep learning models to predict short-term movements in the EUR/USD exchange rate.

Two approaches are implemented and compared:
- Random Forest Classifier (traditional ML baseline)  
- Lightweight LSTM with bidirectional layers and oversampling (deep learning sequence model)  

---

## Features

The dataset is constructed from daily EUR/USD price data (via `yfinance`) with engineered technical indicators and price patterns:

- Returns: 1-day and 5-day returns, normalized returns  
- Candlestick patterns: upper/lower shadows, relative price position  
- Volatility ratio: short-term vs. long-term volatility  
- Target: binary label (1 = next day up, 0 = next day down)  

---

## Models

### Random Forest
- Input: engineered technical features  
- Balanced class weights  
- Achieved ~80% accuracy on the test set  
- Key features:  
  - Lower_shadow (0.477)  
  - Upper_shadow (0.414)  
  - Price_position (0.041)  
  - Return_1d_norm (0.036)  
  - Vol_ratio (0.033)  

### Lightweight LSTM
- Input: rolling sequences of features (5-day windows)  
- Bidirectional LSTM + dropout + gradient clipping  
- Oversampling to reduce class 0 (down days) bias  
- Achieved ~56% accuracy on the test set  
- Better class balance compared to earlier LSTM versions  

---

## Results

- Random Forest remains the stronger performer (~80% accuracy).  
- Lightweight LSTM captures temporal patterns but lags in accuracy (~56%).  
- Demonstrates the challenge of modeling FX time series with deep learning.  

**Note:** Attempts were made to design a trading strategy based on the Random Forest model.  
While predictions showed high accuracy, the strategy delivered **low returns after transaction costs**, so it was ultimately scrapped.   

**Sample trading signals (confidence in brackets):**
2023-12-12: BUY (0.97)
2023-12-15: SELL (0.06)
2023-12-19: BUY (0.98)
2023-12-26: BUY (0.68)
2023-12-28: SELL (0.39)

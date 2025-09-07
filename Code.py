# FX Prediction Model - EUR/USD Exchange Rate
# Predicts next-day direction using Random Forest and LSTM models
# Features: Technical indicators and price patterns

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_fx_data(symbol="EURUSD=X", start="2018-01-01", end="2024-01-01"):
    """Download FX data with proper error handling"""
    try:
        data = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def fix_yfinance_data(df):
    """Fix yfinance data format issues - ensure 1D Series for all columns"""
    df = df.copy()
    
    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Fix each column to be proper 1D Series
    for col in df.columns:
        if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            # Get the values and ensure they're 1D
            values = df[col].values
            if values.ndim > 1:
                values = values.flatten()
            # Create new 1D Series with original index
            df[col] = pd.Series(values, index=df.index)
    
    return df

def create_features(df):
    """Enhanced feature engineering - yfinance data compatible"""
    # Fix data format first
    df = fix_yfinance_data(df)
    
    # Extract price series (now guaranteed to be 1D)
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    open_ = df['Open'] if 'Open' in df.columns else close

    # Basic returns
    df['Return_1d'] = close.pct_change()
    df['Return_3d'] = close.pct_change(3)
    df['Return_5d'] = close.pct_change(5)
    df['Return_10d'] = close.pct_change(10)

    # Normalized returns
    rolling_mean = df['Return_1d'].rolling(20).mean()
    rolling_std = df['Return_1d'].rolling(20).std()
    df['Return_1d_norm'] = (df['Return_1d'] - rolling_mean) / (rolling_std + 1e-8)

    # Moving averages and ratios
    for window in [5, 10, 20, 50]:
        df[f"MA_{window}"] = close.rolling(window).mean()
        df[f"Price_MA_{window}_ratio"] = (close / df[f"MA_{window}"] - 1).fillna(0)

    # RSI - using manual calculation to avoid ta library issues
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = calculate_rsi(close, 14)
    df['RSI_21'] = calculate_rsi(close, 21)
    df['RSI_mean'] = (df['RSI_14'] + df['RSI_21']) / 2
    df['RSI_divergence'] = df['RSI_14'] - df['RSI_21']

    # MACD - using manual calculation
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['MACD_momentum'] = df['MACD_hist'].diff()

    # Bollinger Bands - manual calculation
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_upper'] = sma20 + (std20 * 2)
    df['BB_lower'] = sma20 - (std20 * 2)
    df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (close + 1e-8)
    df['BB_squeeze'] = (df['BB_width'].rolling(10).min() == df['BB_width']).astype(int)

    # ATR - manual calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_norm'] = df['ATR'] / (close + 1e-8)

    # Volatility measures
    df['Vol_5d'] = df['Return_1d'].rolling(5).std() * np.sqrt(252)
    df['Vol_20d'] = df['Return_1d'].rolling(20).std() * np.sqrt(252)
    df['Vol_ratio'] = df['Vol_5d'] / (df['Vol_20d'] + 1e-8)

    # Price patterns
    df['High_Low_ratio'] = (high - low) / (close + 1e-8)
    df['Close_Open_ratio'] = (close - open_) / (open_ + 1e-8)
    df['Upper_shadow'] = (high - np.maximum(close, open_)) / (high - low + 1e-8)
    df['Lower_shadow'] = (np.minimum(close, open_) - low) / (high - low + 1e-8)

    # Trend indicators
    df['MA_5_10_cross'] = (df['MA_5'] > df['MA_10']).astype(int)
    df['MA_10_20_cross'] = (df['MA_10'] > df['MA_20']).astype(int)
    df['Trend_strength'] = (df['MA_5'] - df['MA_50']) / (df['MA_50'] + 1e-8)

    # Support/Resistance
    df['High_20d'] = high.rolling(20).max()
    df['Low_20d'] = low.rolling(20).min()
    df['Price_position'] = (close - df['Low_20d']) / (df['High_20d'] - df['Low_20d'] + 1e-8)

    # Market regime detection
    df['Trending'] = (abs(df['Price_MA_20_ratio']) > 0.02).astype(int)
    df['High_vol'] = (df['Vol_20d'] > df['Vol_20d'].rolling(60).quantile(0.7)).astype(int)

    # Target variable
    df['Target'] = (close.shift(-1) > close).astype(int)

    # Clean up data
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.dropna()

    return df

def select_features(df):
    """Select most predictive features for modeling"""
    feature_cols = [
        'Return_1d_norm', 'Return_5d', 'Return_10d',
        'Price_MA_5_ratio', 'Price_MA_10_ratio', 'Price_MA_20_ratio',
        'RSI_mean', 'RSI_divergence',
        'MACD_hist', 'MACD_momentum',
        'BB_position', 'BB_width', 'BB_squeeze',
        'ATR_norm', 'Vol_ratio',
        'Upper_shadow', 'Lower_shadow',
        'Trend_strength', 'Price_position',
        'Trending', 'High_vol',
        'MA_5_10_cross'
    ]
    available_features = [col for col in feature_cols if col in df.columns]
    return df[available_features], df['Target']

def prepare_data_timeseries(X, y, test_size=0.2):
    """Time series split to avoid look-ahead bias"""
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

def create_sequences(X, y, sequence_length=5):
    """Create sequences for LSTM model"""
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X.iloc[i-sequence_length:i].values)
        y_seq.append(y.iloc[i])
    return np.array(X_seq), np.array(y_seq)

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with reduced overfitting"""
    print("\n=== Random Forest Model ===")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # More conservative RF parameters to reduce overfitting
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)

    # Predictions
    train_preds = rf_model.predict(X_train_scaled)
    test_preds = rf_model.predict(X_test_scaled)
    test_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy: {test_acc:.3f}")
    print(f"Overfitting Gap: {train_acc - test_acc:.3f}")
    
    # Confidence-based predictions
    confident_preds = []
    confident_actual = []
    confidence_threshold = 0.6
    
    for i, prob in enumerate(test_probs):
        if prob > confidence_threshold or prob < (1 - confidence_threshold):
            confident_preds.append(1 if prob > 0.5 else 0)
            confident_actual.append(y_test.iloc[i])
    
    if len(confident_preds) > 0:
        confident_acc = accuracy_score(confident_actual, confident_preds)
        print(f"Confident Predictions ({len(confident_preds)}/{len(y_test)}): {confident_acc:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

    return rf_model, scaler, test_acc, feature_importance

def train_lstm(X_train, y_train, X_test, y_test, sequence_length=5):
    """Train LSTM model with improved architecture"""
    print("\n=== LSTM Model ===")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(
        pd.DataFrame(X_train_scaled, columns=X_train.columns), 
        pd.Series(y_train.values), 
        sequence_length
    )
    X_test_seq, y_test_seq = create_sequences(
        pd.DataFrame(X_test_scaled, columns=X_test.columns), 
        pd.Series(y_test.values), 
        sequence_length
    )

    print(f"LSTM input shape: {X_train_seq.shape}")

    # Simplified LSTM architecture
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])),
        Dropout(0.3),
        LSTM(16, return_sequences=False),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6
    )

    # Train model
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_seq, y_test_seq),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Predictions
    train_preds = model.predict(X_train_seq)
    test_preds = model.predict(X_test_seq)
    
    train_preds_binary = (train_preds > 0.5).astype(int).flatten()
    test_preds_binary = (test_preds > 0.5).astype(int).flatten()
    
    train_acc = accuracy_score(y_train_seq, train_preds_binary)
    test_acc = accuracy_score(y_test_seq, test_preds_binary)

    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy: {test_acc:.3f}")
    print(f"Overfitting Gap: {train_acc - test_acc:.3f}")
    print(f"Epochs trained: {len(history.history['loss'])}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_seq, test_preds_binary))

    return model, scaler, test_acc, history

def generate_trading_signals(model, scaler, X_recent, threshold=0.6):
    """Generate trading signals with confidence threshold"""
    X_scaled = scaler.transform(X_recent)
    
    if hasattr(model, 'predict_proba'):  # Random Forest
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:  # LSTM
        probabilities = model.predict(X_scaled).flatten()
    
    signals = []
    for prob in probabilities:
        if prob > threshold:
            signals.append('BUY')
        elif prob < (1 - threshold):
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals, probabilities

def main():
    """Main execution pipeline"""
    print("FX Prediction Model - EUR/USD")
    print("=" * 40)
    
    try:
        # 1. Data Collection
        print("Downloading EUR/USD data...")
        df = get_fx_data()
        if df is None:
            print("Failed to download data. Exiting.")
            return
        
        print(f"Raw data shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # 2. Feature Engineering
        print("\nCreating features...")
        df_features = create_features(df)
        print(f"Features created successfully. Shape: {df_features.shape}")
        
        X, y = select_features(df_features)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = prepare_data_timeseries(X, y, test_size=0.2)
        
        print(f"\nTrain period: {X_train.index[0]} to {X_train.index[-1]}")
        print(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
        
        # 4. Train Models
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        rf_model, rf_scaler, rf_accuracy, feature_importance = train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        lstm_model, lstm_scaler, lstm_accuracy, lstm_history = train_lstm(
            X_train, y_train, X_test, y_test, sequence_length=5
        )
        
        # 5. Model Comparison
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(f"Random Forest Accuracy: {rf_accuracy:.1%}")
        print(f"LSTM Accuracy: {lstm_accuracy:.1%}")
        
        best_model = rf_model if rf_accuracy > lstm_accuracy else lstm_model
        best_scaler = rf_scaler if rf_accuracy > lstm_accuracy else lstm_scaler
        best_accuracy = max(rf_accuracy, lstm_accuracy)
        
        # 6. Recent Signals
        print("\n" + "="*50)
        print("RECENT TRADING SIGNALS")
        print("="*50)
        
        X_recent = X_test.tail(5)
        signals, probabilities = generate_trading_signals(best_model, best_scaler, X_recent)
        
        for i, (signal, prob) in enumerate(zip(signals, probabilities)):
            date = X_recent.index[i]
            print(f"{date.strftime('%Y-%m-%d')}: {signal} (confidence: {prob:.2f})")
        
        # 7. Summary
        print("\n" + "="*50)
        print("PROJECT SUMMARY")
        print("="*50)
        print("• Developed ML models (Random Forest, LSTM) for EUR/USD prediction")
        print("• Incorporated 22 enhanced technical analysis features")
        print(f"• Achieved {best_accuracy:.1%} directional accuracy")
        print("• Implemented overfitting reduction techniques")
        print("• Added confidence-based signal generation")
        print("• Manual indicator calculations for data compatibility")
        
        if rf_accuracy >= lstm_accuracy:
            print(f"\nTop 5 Most Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
    except Exception as e:
        print(f"Error in execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

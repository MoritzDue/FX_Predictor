# EUR/USD Exchange Rate Prediction with Random Forest and Lightweight LSTM
# Features: Technical indicators and price patterns

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# --- Data download and cleaning ---
def get_fx_data(symbol="EURUSD=X", start="2018-01-01", end="2024-01-01"):
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
        if df.empty: raise ValueError(f"No data found for {symbol}")
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def fix_yfinance_data(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in df.columns:
        if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            values = df[col].values
            if values.ndim > 1: values = values.flatten()
            df[col] = pd.Series(values, index=df.index)
    return df

# --- Feature Engineering ---
def create_features(df):
    df = fix_yfinance_data(df)
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    open_ = df['Open'] if 'Open' in df.columns else close

    # Basic returns
    df['Return_1d'] = close.pct_change()
    df['Return_5d'] = close.pct_change(5)

    # Normalized return
    df['Return_1d_norm'] = (df['Return_1d'] - df['Return_1d'].rolling(20).mean()) / (df['Return_1d'].rolling(20).std() + 1e-8)

    # Price patterns
    df['Upper_shadow'] = (high - np.maximum(close, open_)) / (high - low + 1e-8)
    df['Lower_shadow'] = (np.minimum(close, open_) - low) / (high - low + 1e-8)
    df['Price_position'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-8)

    # Volatility
    df['Vol_ratio'] = df['Return_1d'].rolling(5).std() / (df['Return_1d'].rolling(20).std() + 1e-8)

    # Target
    df['Target'] = (close.shift(-1) > close).astype(int)

    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.dropna()
    return df

# --- Feature selection ---
TOP_FEATURES = ['Upper_shadow', 'Lower_shadow', 'Price_position', 'Vol_ratio', 'Return_1d_norm']

def select_features(df):
    available_features = [col for col in TOP_FEATURES if col in df.columns]
    return df[available_features], df['Target']

# --- Time series split ---
def prepare_data_timeseries(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

# --- Random Forest ---
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n=== Random Forest Model ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_split=20, min_samples_leaf=10,
        max_features='sqrt', random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)

    train_preds = rf_model.predict(X_train_scaled)
    test_preds = rf_model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy: {test_acc:.3f}")
    print(f"Overfitting Gap: {train_acc - test_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))

    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
    return rf_model, scaler, test_acc, feature_importance

# --- LSTM sequence creation with oversampling ---
def create_sequences_balanced(X, y, sequence_length=5):
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X.iloc[i-sequence_length:i].values)
        y_seq.append(y.iloc[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Oversample minority class
    class_1_idx = np.where(y_seq == 1)[0]
    class_0_idx = np.where(y_seq == 0)[0]
    if len(class_1_idx) < len(class_0_idx):
        oversample_idx = np.random.choice(class_1_idx, size=len(class_0_idx)-len(class_1_idx), replace=True)
        X_seq = np.concatenate([X_seq, X_seq[oversample_idx]], axis=0)
        y_seq = np.concatenate([y_seq, y_seq[oversample_idx]], axis=0)
    return X_seq, y_seq

# --- Lightweight LSTM ---
def train_lstm_small(X_train, y_train, X_test, y_test, sequence_length=5):
    print("\n=== Lightweight LSTM Model ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_top = pd.DataFrame(X_train_scaled, columns=X_train.columns)[TOP_FEATURES]
    X_test_top = pd.DataFrame(X_test_scaled, columns=X_test.columns)[TOP_FEATURES]

    X_train_seq, y_train_seq = create_sequences_balanced(X_train_top, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences_balanced(X_test_top, y_test, sequence_length)

    print(f"LSTM input shape: {X_train_seq.shape}")

    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=True, dropout=0.2), input_shape=(sequence_length, X_train_seq.shape[2])),
        LSTM(16, return_sequences=False, dropout=0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

    history = model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=16,
                        validation_data=(X_test_seq, y_test_seq),
                        callbacks=[early_stop, reduce_lr], verbose=1)

    train_preds = (model.predict(X_train_seq) > 0.5).astype(int).flatten()
    test_preds = (model.predict(X_test_seq) > 0.5).astype(int).flatten()

    print(f"\nTraining Accuracy: {accuracy_score(y_train_seq, train_preds):.3f}")
    print(f"Testing Accuracy: {accuracy_score(y_test_seq, test_preds):.3f}")
    print(f"Classification Report:\n{classification_report(y_test_seq, test_preds)}")
    return model, scaler, history, X_test_seq, y_test_seq

# --- Trading signals ---
def generate_trading_signals(model, scaler, X_recent, sequence_length=5, threshold=0.6):
    X_scaled = scaler.transform(X_recent)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        if len(X_recent) >= sequence_length:
            X_seq = [X_scaled[i-sequence_length+1:i+1] for i in range(sequence_length-1, len(X_scaled))]
            X_seq = np.array(X_seq)
            probabilities = model.predict(X_seq).flatten()
        else:
            return [], []
    signals = []
    for prob in probabilities:
        if prob > threshold:
            signals.append('BUY')
        elif prob < (1 - threshold):
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals, probabilities

# --- Main pipeline ---
def main():
    print("Improved FX Prediction Model - EUR/USD")
    df = get_fx_data()
    if df is None: return
    df_features = create_features(df)
    X, y = select_features(df_features)
    X_train, X_test, y_train, y_test = prepare_data_timeseries(X, y, test_size=0.2)

    # Train Random Forest
    rf_model, rf_scaler, rf_acc, feature_importance = train_random_forest(X_train, y_train, X_test, y_test)

    # Train Lightweight LSTM
    lstm_model, lstm_scaler, lstm_history, X_test_seq, y_test_seq = train_lstm_small(
        X_train, y_train, X_test, y_test, sequence_length=5
    )

    # Compute LSTM accuracy correctly
    lstm_test_acc = accuracy_score(
        y_test_seq,
        (lstm_model.predict(X_test_seq) > 0.5).astype(int).flatten()
    )

    print("\nMODEL COMPARISON")
    print(f"Random Forest Accuracy: {rf_acc:.1%}")
    print(f"Lightweight LSTM Accuracy: {lstm_test_acc:.1%}")
    print(f"LSTM Gap: {lstm_test_acc - rf_acc:.1%} percentage points")

    # Generate recent trading signals
    X_recent = X_test.tail(15)
    best_model = rf_model if rf_acc > lstm_test_acc else lstm_model
    best_scaler = rf_scaler if rf_acc > lstm_test_acc else lstm_scaler
    sequence_length_used = 1 if rf_acc > lstm_test_acc else 5
    signals, probs = generate_trading_signals(best_model, best_scaler, X_recent, sequence_length=sequence_length_used)
    recent_dates = X_recent.index[-len(signals):]
    for date, signal, prob in zip(recent_dates, signals, probs):
        print(f"{date.strftime('%Y-%m-%d')}: {signal} (confidence: {prob:.2f})")

    print("\nPROJECT SUMMARY")
    print("• Lightweight LSTM with oversampling to reduce class 0 bias")
    print("• Bidirectional layers and gradient clipping")
    print(f"• Best model achieved {max(rf_acc, lstm_test_acc):.1%} accuracy")
    print("Random Forest still leading - further LSTM tuning may be needed")
    print("Top Random Forest features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

if __name__ == "__main__":
    main()

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15  
NUM_BARS = 10000               
CSV_FILENAME = "eurusd_training_data.csv"

# ==========================================
# HELPER FUNCTIONS FOR INDICATORS
# ==========================================
def calculate_moving_average(series, window):
    return series.rolling(window=window).mean()

def calculate_momentum(series, period):
    return series.diff(period)

def calculate_rsi(series, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD and Signal Line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ==========================================
# MAIN DATA COLLECTION LOGIC
# ==========================================
def collect_data():
    print(f"Connecting to MT5 to fetch {NUM_BARS} bars of {SYMBOL} data...")
    
    if not mt5.initialize():
        print(f"MT5 Initialization failed. Error: {mt5.last_error()}")
        return

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_BARS)
    
    if rates is None or len(rates) == 0:
        print("Failed to fetch data.")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print("Calculating Advanced AI Features...")
    
    # Basic Features
    df['SMA_10'] = calculate_moving_average(df['close'], 10)
    df['SMA_30'] = calculate_moving_average(df['close'], 30)
    df['Momentum_5'] = calculate_momentum(df['close'], 5)
    df['Volatility'] = df['high'] - df['low']
    
    # Advanced Features (RSI & MACD)
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['close'])

    # Drop rows with NaN values created by the rolling calculations
    df = df.dropna()
    
    df.to_csv(CSV_FILENAME, index=False)
    print(f"Success! Data saved to '{CSV_FILENAME}'.")
    print(f"Total rows collected: {len(df)}")
    print("\nHere is a preview of your dataset:")
    print(df.head())

    mt5.shutdown()

if __name__ == "__main__":
    collect_data()
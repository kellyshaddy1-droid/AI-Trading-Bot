import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "EURUSD"           
TIMEFRAME = mt5.TIMEFRAME_M15 
LOT_SIZE = 0.01             
MAGIC_NUMBER = 234000       
SLIPPAGE = 20               
CONFIDENCE_THRESHOLD = 0.60 # ONLY TRADE IF AI IS >60% SURE

# Load the V4 AI Model
print("Loading V4 Short-Term AI Model...")
try:
    ai_model = joblib.load('eurusd_ai_model_v4.pkl')
    print("AI Model loaded successfully!")
except FileNotFoundError:
    print("Error: eurusd_ai_model_v4.pkl not found. Run ai_trainer_v4.py first.")
    exit()

# The exact features the V4 model was trained on
AI_FEATURES = ['tick_volume', 'Momentum_5', 'Volatility', 'RSI', 'MACD', 'Signal_Line']

# ==========================================
# INDICATOR CALCULATIONS
# ==========================================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def initialize_mt5():
    if not mt5.initialize():
        print(f"MT5 Initialization failed. Error code: {mt5.last_error()}")
        return False
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"Symbol {SYMBOL} not found.")
        mt5.shutdown()
        return False
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            mt5.shutdown()
            return False
    print(f"Successfully connected to MT5. Account: {mt5.account_info().login}")
    return True

def get_market_data():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate short-term features
    df['Momentum_5'] = df['close'].diff(5)
    df['Volatility'] = df['high'] - df['low']
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['close'])
    
    df = df.dropna()
    return df

def generate_signal(df):
    if df is None or len(df) < 20:
        return "HOLD", 0.0
        
    latest_data = df[AI_FEATURES].tail(1)
    
    # Get prediction AND probability
    prediction = ai_model.predict(latest_data)[0]
    probabilities = ai_model.predict_proba(latest_data)[0]
    
    # Probabilities[0] is chance of Down, [1] is chance of Up
    if prediction == 1:
        confidence = probabilities[1]
    else:
        confidence = probabilities[0]
        
    print(f"   -> AI Prediction: {'UP' if prediction==1 else 'DOWN'} | Confidence: {confidence*100:.1f}%")
    
    # THE FILTER: Only trade if confidence is high enough
    if confidence < CONFIDENCE_THRESHOLD:
        print("   -> Confidence too low. Holding.")
        return "HOLD", confidence
        
    if prediction == 1:
        return "BUY", confidence
    else:
        return "SELL", confidence

def execute_trade(signal):
    symbol_info = mt5.symbol_info(SYMBOL)
    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)
        
    point = symbol_info.point
    price = mt5.symbol_info_tick(SYMBOL).ask if signal == "BUY" else mt5.symbol_info_tick(SYMBOL).bid
    
    # Standard 50-point Stop Loss / Take Profit for 15-min trades
    sl_tp_distance = 50 * point 
    stop_loss = price - sl_tp_distance if signal == "BUY" else price + sl_tp_distance
    take_profit = price + sl_tp_distance if signal == "BUY" else price - sl_tp_distance

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": "V4 AI Bot 15m",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed. Retcode: {result.retcode}")
    else:
        print(f"Success! {signal} order executed. Ticket: {result.order}")

def main():
    print("Starting V4 AI Trading Bot (15-Minute High Confidence Mode)...")
    if not initialize_mt5():
        return

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking market...")
            data = get_market_data()
            signal, conf = generate_signal(data)
            
            if signal in ["BUY", "SELL"]:
                current_positions = mt5.positions_get(symbol=SYMBOL)
                has_open_position = any(p.magic == MAGIC_NUMBER for p in current_positions) if current_positions else False
                
                if not has_open_position:
                    execute_trade(signal)
                else:
                    print("   -> Position already open. Waiting.")
                    
            time.sleep(60) # Check every 60 seconds
    except KeyboardInterrupt:
        print("\nBot stopped.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
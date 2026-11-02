import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# The pairs we want to scan
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"] 
TIMEFRAME = mt5.TIMEFRAME_M15 
LOT_SIZE = 0.01             
MAGIC_NUMBER = 234000       
SLIPPAGE = 20               
CONFIDENCE_THRESHOLD = 0.60 # Only trade if confidence is > 60%

# Load the AI Model
print("Loading AI Model...")
try:
    ai_model = joblib.load('eurusd_ai_model_v4.pkl')
    print("AI Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model not found. Run ai_trainer_v4.py first.")
    exit()

# Features the AI expects
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

# ==========================================
# INITIALIZATION
# ==========================================
def initialize_mt5():
    if not mt5.initialize():
        print(f"MT5 Initialization failed. Error code: {mt5.last_error()}")
        return False
    
    # Ensure all symbols are visible in Market Watch
    for sym in SYMBOLS:
        symbol_info = mt5.symbol_info(sym)
        if symbol_info is None:
            print(f"Symbol {sym} not found.")
            continue
        if not symbol_info.visible:
            print(f"Enabling {sym} in Market Watch...")
            mt5.symbol_select(sym, True)
            
    print(f"Successfully connected to MT5. Account: {mt5.account_info().login}")
    return True

# ==========================================
# SCANNING LOGIC
# ==========================================
def scan_market():
    """Scans all pairs and returns the single best trade opportunity."""
    opportunities = []
    
    for sym in SYMBOLS:
        rates = mt5.copy_rates_from_pos(sym, TIMEFRAME, 0, 100)
        if rates is None or len(rates) == 0:
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate features
        df['Momentum_5'] = df['close'].diff(5)
        df['Volatility'] = df['high'] - df['low']
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['Signal_Line'] = calculate_macd(df['close'])
        df = df.dropna()
        
        if len(df) < 20:
            continue
            
        latest_data = df[AI_FEATURES].tail(1)
        
        # Get AI prediction and confidence
        prediction = ai_model.predict(latest_data)[0]
        probabilities = ai_model.predict_proba(latest_data)[0]
        
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        signal = "BUY" if prediction == 1 else "SELL"
        
        print(f"  [{sym}] {signal} | Confidence: {confidence*100:.1f}%")
        
        # If it meets our threshold, add it to the list of opportunities
        if confidence >= CONFIDENCE_THRESHOLD:
            opportunities.append({
                'symbol': sym,
                'signal': signal,
                'confidence': confidence
            })
            
    # Sort opportunities by confidence (Highest first)
    opportunities.sort(key=lambda x: x['confidence'], reverse=True)
    
    if len(opportunities) > 0:
        return opportunities[0] # Return the absolute best one
    else:
        return None

# ==========================================
# TRADE EXECUTION
# ==========================================
def execute_trade(opportunity):
    sym = opportunity['symbol']
    signal = opportunity['signal']
    
    symbol_info = mt5.symbol_info(sym)
    if not symbol_info.visible:
        mt5.symbol_select(sym, True)
        
    point = symbol_info.point
    price = mt5.symbol_info_tick(sym).ask if signal == "BUY" else mt5.symbol_info_tick(sym).bid
    
    # Dynamic Stop Loss / Take Profit (100 points)
    sl_tp_distance = 100 * point 
    stop_loss = price - sl_tp_distance if signal == "BUY" else price + sl_tp_distance
    take_profit = price + sl_tp_distance if signal == "BUY" else price - sl_tp_distance

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": "Scanner Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed on {sym}. Retcode: {result.retcode}")
    else:
        print(f"\n🚀 EXECUTED {signal} on {sym}! (Confidence: {opportunity['confidence']*100:.1f}%)")

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    print("Starting Multi-Pair Scanner Bot...")
    if not initialize_mt5():
        return

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning Market...")
            
            best_trade = scan_market()
            
            if best_trade:
                print(f"\n🎯 Best Opportunity Found: {best_trade['symbol']} ({best_trade['signal']})")
                
                # Check if we already have a position in this specific pair
                positions = mt5.positions_get(symbol=best_trade['symbol'])
                has_open_position = any(p.magic == MAGIC_NUMBER for p in positions) if positions else False
                
                if not has_open_position:
                    execute_trade(best_trade)
                else:
                    print(f"   -> Already have a position in {best_trade['symbol']}. Skipping.")
            else:
                print("No high-confidence setups found. Waiting...")
                
            # Wait 60 seconds before scanning again
            time.sleep(60) 
            
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
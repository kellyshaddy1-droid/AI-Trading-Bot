import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Trading Bot", layout="wide", page_icon="📈")
st.title("🤖 AI Trading Bot Dashboard")

# ==========================================
# SESSION STATE (Remembers variables between reruns)
# ==========================================
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = "Waiting..."
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Control Panel")

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶️ Start Bot", use_container_width=True)
stop_btn = col2.button("⏹️ Stop Bot", use_container_width=True)

if start_btn:
    st.session_state.bot_running = True
if stop_btn:
    st.session_state.bot_running = False

status_color = "🟢" if st.session_state.bot_running else "🔴"
status_text = "Running" if st.session_state.bot_running else "Stopped"
st.sidebar.markdown(f"**Status:** {status_color} {status_text}")

st.sidebar.markdown("---")
st.sidebar.subheader("Account Info")

# ==========================================
# MT5 CONNECTION & AI LOADING
# ==========================================
@st.cache_resource # Caches the connection so it doesn't reconnect every second
def connect_mt5():
    if not mt5.initialize():
        return None
    return mt5.account_info().login

@st.cache_resource # Caches the model so it doesn't reload from disk every second
def load_ai_model():
    return joblib.load('eurusd_ai_model.pkl')

account_login = connect_mt5()
if account_login:
    st.sidebar.success(f"Connected to MT5\nAccount: {account_login}")
else:
    st.sidebar.error("Failed to connect to MT5. Is the terminal open?")
    st.stop() # Stop the app if MT5 isn't connected

try:
    ai_model = load_ai_model()
except FileNotFoundError:
    st.sidebar.error("AI Model not found! Run ai_trainer.py first.")
    st.stop()

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
# MAIN DASHBOARD LAYOUT
# ==========================================
chart_col, signal_col = st.columns(2)

# --- LEFT COLUMN: LIVE CHART ---
with chart_col:
    st.subheader("Live EURUSD Market Data (M15)")
    
    # Fetch latest 100 candles
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 100)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Display chart
    st.line_chart(df.set_index('time')['close'], use_container_width=True)
    
    # Display current price metrics
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    change = current_price - prev_price
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"{current_price:.5f}", f"{change:.5f}")
    m2.metric("Daily High", f"{df['high'].max():.5f}")
    m3.metric("Daily Low", f"{df['low'].min():.5f}")

# --- RIGHT COLUMN: AI ANALYSIS ---
with signal_col:
    st.subheader("AI Model Analysis")
    
    # Calculate indicators for the AI
    df['Momentum_5'] = df['close'].diff(5)
    df['Volatility'] = df['high'] - df['low']
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['close'])
    df = df.dropna()
    
    if len(df) > 0:
        latest = df.iloc[-1]
        
        # Show Indicator Metrics
        rsi_val = latest['RSI']
        macd_val = latest['MACD']
        
        r1, r2 = st.columns(2)
        r1.metric("RSI (14)", f"{rsi_val:.2f}", help=">70 Overbought, <30 Oversold")
        r2.metric("MACD", f"{macd_val:.5f}")
        
        # AI Prediction
        AI_FEATURES = ['tick_volume', 'real_volume', 'Momentum_5', 'Volatility', 'RSI', 'MACD', 'Signal_Line']
        features_df = df[AI_FEATURES].tail(1)
        
        prediction = ai_model.predict(features_df)[0]
        probabilities = ai_model.predict_proba(features_df)[0]
        
        signal = "BUY 🟢" if prediction == 1 else "SELL 🔴"
        confidence = max(probabilities) * 100
        
        st.markdown(f"### AI Decision: {signal}")
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence:.1f}% | Down: {probabilities[0]*100:.1f}% | Up: {probabilities[1]*100:.1f}%")

# ==========================================
# BOT EXECUTION LOGIC (The Loop)
# ==========================================
if st.session_state.bot_running:
    st.markdown("---")
    st.subheader("🔄 Bot Execution Log")
    
    # Execute trade logic if signal is strong
    if signal in ["BUY 🟢", "SELL 🔴"]:
        # Check if we already have a position
        positions = mt5.positions_get(symbol="EURUSD")
        has_position = False
        if positions:
            for pos in positions:
                if pos.magic == 234000:
                    has_position = True
                    break
        
        if not has_position:
            # Prepare Order
            symbol_info = mt5.symbol_info("EURUSD")
            point = symbol_info.point
            price = mt5.symbol_info_tick("EURUSD").ask if prediction == 1 else mt5.symbol_info_tick("EURUSD").bid
            order_type = mt5.ORDER_TYPE_BUY if prediction == 1 else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "EURUSD",
                "volume": 0.01,
                "type": order_type,
                "price": price,
                "sl": price - (50 * point) if prediction == 1 else price + (50 * point),
                "tp": price + (50 * point) if prediction == 1 else price - (50 * point),
                "deviation": 20,
                "magic": 234000,
                "comment": "Streamlit AI Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Executed {signal} at {price:.5f} (Ticket: {result.order})"
                st.session_state.trade_log.insert(0, log_msg) # Add to top
            else:
                log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Order Failed: {result.retcode}"
                st.session_state.trade_log.insert(0, log_msg)
        else:
            log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ⏸️ Signal: {signal}, but position already open. Holding."
            # Only add log if it's different from the last one to prevent spam
            if not st.session_state.trade_log or "position already open" not in st.session_state.trade_log[0]:
                 st.session_state.trade_log.insert(0, log_msg)

    # Display the log
    for log in st.session_state.trade_log[:5]: # Show last 5 logs
        st.text(log)
        
    # THE STREAMLIT LOOP TRICK:
    # Wait 15 seconds, then rerun the whole script to check the market again.
    time.sleep(15)
    st.rerun()

else:
    st.info("Bot is stopped. Click 'Start Bot' in the sidebar to begin automated trading.")
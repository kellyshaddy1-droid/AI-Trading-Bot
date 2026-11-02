import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier # Smarter algorithm
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading data...")
df = pd.read_csv('eurusd_training_data.csv')

print("Creating Advanced 4-Hour Features...")
# We calculate features that cover the full 4-hour (16 candle) window
df['Momentum_16'] = df['close'] - df['close'].shift(16) # Price change over 4 hours
df['Volatility_16'] = df['high'].rolling(16).max() - df['low'].rolling(16).min() # Range over 4 hours

# Target: Will price be higher in 16 candles (4 hours)?
FUTURE_CANDLES = 16
df['future_price'] = df['close'].shift(-FUTURE_CANDLES)
df['target'] = (df['future_price'] > df['close']).astype(int)

# Drop rows where we couldn't calculate the 16-candle features
df = df.dropna()

print(f"Class distribution:\n{df['target'].value_counts()}")

# Define Features (X) and Target (y)
# We removed 'real_volume' because your last report showed it had 0 importance!
# We added Momentum_16 and Volatility_16
features = ['tick_volume', 'Momentum_5', 'Momentum_16', 'Volatility', 'Volatility_16', 'RSI', 'MACD', 'Signal_Line']

X = df[features]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("\nTraining Gradient Boosting AI Model...")
# Gradient Boosting builds trees sequentially to fix mistakes
model = GradientBoostingClassifier(
    n_estimators=200, 
    learning_rate=0.05, # Slower learning = smarter model
    max_depth=5, 
    random_state=42
)

model.fit(X_train, y_train)

print("\nTesting the AI...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print(f" V3 AI MODEL TRAINING COMPLETE!")
print(f"📊 Accuracy: {accuracy * 100:.2f}%")
print(f"🎯 Predicting 4 hours ahead with aligned features")
print("="*50 + "\n")

print("Detailed Report:")
print(classification_report(y_test, predictions, target_names=['Down (0)', 'Up (1)']))

joblib.dump(model, 'eurusd_ai_model_v3.pkl')
print("✅ Model saved to 'eurusd_ai_model_v3.pkl'")
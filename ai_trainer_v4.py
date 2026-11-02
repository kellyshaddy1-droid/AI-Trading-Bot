import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading data...")
df = pd.read_csv('eurusd_training_data.csv')

print("Creating AI Target (Predicting NEXT 15-min candle)...")
# Target: 1 if the NEXT candle's close is higher, 0 if it's lower
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()

# Define Features (Short-term focused)
features = ['tick_volume', 'Momentum_5', 'Volatility', 'RSI', 'MACD', 'Signal_Line']

X = df[features]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("\nTraining Short-Term AI Model...")
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    class_weight='balanced', # Crucial for short-term to prevent lazy "always up" predictions
    random_state=42
)

model.fit(X_train, y_train)

print("\nTesting the AI...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print(f" V4 SHORT-TERM AI MODEL COMPLETE!")
print(f"📊 Accuracy: {accuracy * 100:.2f}%")
print(f"🎯 Predicting NEXT 15-minute candle")
print("="*50 + "\n")

print("Detailed Report:")
print(classification_report(y_test, predictions, target_names=['Down (0)', 'Up (1)']))

joblib.dump(model, 'eurusd_ai_model_v4.pkl')
print("✅ Model saved to 'eurusd_ai_model_v4.pkl'")
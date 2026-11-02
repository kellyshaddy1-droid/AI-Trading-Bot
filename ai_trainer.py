import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading data...")
df = pd.read_csv('eurusd_training_data.csv')

print("Creating AI Target (Predicting 4 hours into the future)...")
# Instead of predicting the NEXT candle (shift(-1)), we predict 16 candles ahead
# 16 candles × 15 minutes = 240 minutes = 4 hours
FUTURE_CANDLES = 16

df['future_price'] = df['close'].shift(-FUTURE_CANDLES)
df['target'] = (df['future_price'] > df['close']).astype(int)

# Drop the last FUTURE_CANDLES rows because they don't have a future price to compare to
df = df.iloc[:-FUTURE_CANDLES]
df = df.dropna()

print(f"Target created: Will price be higher in {FUTURE_CANDLES} candles (4 hours)?")
print(f"Class distribution:\n{df['target'].value_counts()}")

# Define Features (X) and Target (y)
features = ['tick_volume', 'real_volume', 'Momentum_5', 'Volatility', 'RSI', 'MACD', 'Signal_Line']

X = df[features]
y = df['target']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("\nTraining the Random Forest AI Model...")
print("(This may take a minute with the new parameters)")

# Train with balanced classes and more trees for better pattern recognition
model = RandomForestClassifier(
    n_estimators=300,        # More trees = smarter model
    max_depth=15,            # Deeper trees can find more complex patterns
    min_samples_split=10,    # Prevent overfitting
    min_samples_leaf=5,
    class_weight='balanced', # Force it to learn both UP and DOWN patterns
    random_state=42,
    n_jobs=-1                # Use all CPU cores for faster training
)

model.fit(X_train, y_train)

print("\nTesting the AI on unseen data...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print(f" UPGRADED AI MODEL TRAINING COMPLETE!")
print(f"📊 Accuracy on unseen data: {accuracy * 100:.2f}%")
print(f"🎯 Predicting {FUTURE_CANDLES} candles ahead (4 hours)")
print("="*50 + "\n")

print("Detailed Classification Report:")
print(classification_report(y_test, predictions, target_names=['Down in 4hrs (0)', 'Up in 4hrs (1)']))

# Show feature importance (which indicators matter most?)
print("\n📈 Feature Importance (What the AI cares about most):")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in feature_importance.iterrows():
    print(f"  {row['Feature']:15} {row['Importance']:.3f}")

joblib.dump(model, 'eurusd_ai_model_v2.pkl')
print("\n✅ Upgraded model saved to 'eurusd_ai_model_v2.pkl'")
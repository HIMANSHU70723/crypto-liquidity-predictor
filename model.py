import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load cleaned dataset (make sure this file exists in the 'data/' folder)
df = pd.read_csv("data/crypto_clean.csv")

# Feature engineering
df['ma_7'] = df['price'].rolling(window=7).mean()
df['volatility'] = df['price'].rolling(window=7).std()
df['liquidity_ratio'] = df['24h_volume'] / (df['volatility'] + 1e-6)
df.dropna(inplace=True)

# Features and target
X = df[['24h_volume', 'ma_7', 'volatility', 'liquidity_ratio']]
y = df['liquidity']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "liquidity_model.pkl")
print("Model trained and saved as 'liquidity_model.pkl'")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from data_preprocessing import preprocess_data

# Load and preprocess the data
df = pd.read_csv(r"C:\Users\Sumay\Desktop\Yameen\Codesoft\DataScience\Movie rating\IMDb Movies India.csv\IMDb Movies India.csv", encoding='latin1')
df = preprocess_data(df)

# Select features and target
features = [col for col in df.columns if col not in ['Name', 'Rating']]
target = 'Rating'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

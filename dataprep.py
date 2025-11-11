# data prep for 70-15-15 train-validation-test-split
# pip install pandas scikit-learn before running

import pandas as pd
from sklearn.model_selection import train_test_split
from features import extract_features  # Import from our new features.py file

# Read the CSV file
df = pd.read_csv('annotations.csv')
print("Dataset loaded. Shape:", df.shape)

# Drop rows with missing words or labels
df = df.dropna(subset=['word', 'corrected_label'])
print(f"After removing missing values: {df.shape}")

# Convert word column to string (just in case may mixed types)
df['word'] = df['word'].astype(str)

# Extract features for each word using the function from features.py
X = [extract_features(word) for word in df["word"]]
y = df["corrected_label"].tolist()

print(f"Extracted {len(X)} feature vectors with {len(X[0])} features each")

# note that param. random_state can be any integer
# Split dataset 70% test, 30% temp data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=67, test_size=0.3)
# Split temp set, 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=67, test_size=0.5)

# check if data was split correctly
# note that the values are rounded up to integers
# Train: 70% of 52 words = 36.4 -> 37
print("Train:", len(X_train))
# Temporary: 30% of 52 = 15.6 -> 16
# Validation: 15% of 52 = 7.8 -> 8 
print("Validation:", len(X_val))
# Test: 15% of 52 = 7.8 -> 8
print("Test:", len(X_test))
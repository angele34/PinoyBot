# data prep for 70-15-15 train-validation-test-split
# pip install pandas scikit-learn before running

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('annotations.csv')
# print(df.head())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["word"])
y = df["label"]

# note that param. random_state can be any integer
# Split dataset 70% test, 30% temp data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=67, test_size=0.3)
# Split temp set, 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=67, test_size=0.5)

# check if data was split correctly
# note that the values are rounded up to integers
# Train: 70% of 52 words = 36.4 -> 37
print("Train:", X_train.shape)
# Temporary: 30% of 52 = 15.6 -> 16
# Validation: 15% of 52 = 7.8 -> 8 
print("Validation:", X_val.shape)
# Test: 15% of 52 = 7.8 -> 8
print("Test:", X_test.shape)
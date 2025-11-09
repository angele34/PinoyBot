# Train data using Naive Bayes and Decision Tree
# Run using python classifier.py py
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from dataprep import *

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
print("Naive Bayes Accuracy:", nb.score(X_test, y_test)*100)

# Decision Tree
dt = DecisionTreeClassifier(random_state=67)
dt.fit(X_train, y_train)
print("Decision Tree Accuracy:", dt.score(X_test, y_test)*100)
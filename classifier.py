# Train data using Naive Bayes and Decision Tree
# Run using python classifier.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from dataprep import *
import pickle

print("\n\nTraining Models\n")

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_accuracy = nb.score(X_test, y_test) * 100
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}%")

# Decision Tree
dt = DecisionTreeClassifier(random_state=67)
dt.fit(X_train, y_train)
dt_accuracy = dt.score(X_test, y_test) * 100
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}%")

# Choose more accurate model
if dt_accuracy >= nb_accuracy:
    best_model = dt
    model_name = "Decision Tree"
else:
    best_model = nb
    model_name = "Naive Bayes"

# Save best model
with open('language_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
print(f"\nMost accurate model: ({model_name}) saved as 'language_model.pkl'")
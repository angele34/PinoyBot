from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from dataprep import *
import pickle

print("\n\nTraining Models\n")

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_accuracy = nb.score(X_test, y_test) * 100
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}%")

# Decision Tree with Grid Search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced'],
    'ccp_alpha': [0.0, 0.001, 0.01], 
}

dt_base = DecisionTreeClassifier(random_state=67)
grid_search = GridSearchCV(dt_base, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest Decision Tree parameters: {grid_search.best_params_}")
dt = grid_search.best_estimator_

dt_accuracy = dt.score(X_test, y_test) * 100

y_pred = dt.predict(X_test)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred))

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
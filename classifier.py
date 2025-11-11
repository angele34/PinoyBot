from dataprep import *
import pickle
from algorithms import train_naive_bayes, train_decision_tree, train_random_forest

print("\n\nTraining Models\n")

# Train Naive Bayes
nb, nb_accuracy, nb_f1, y_pred_nb = train_naive_bayes(X_train, y_train, X_test, y_test)

# Train Decision Tree with Grid Search
dt, dt_accuracy, dt_f1, y_pred_dt = train_decision_tree(
    X_train, y_train, X_test, y_test, X_val, y_val
)

# Train Random Forest with Grid Search
rf, rf_accuracy, rf_f1, y_pred_rf = train_random_forest(
    X_train, y_train, X_test, y_test, X_val, y_val
)

# Model Comparison
print("\n--- Model Comparison ---")
print(f"Decision Tree F1-macro: {dt_f1:.4f}")
print(f"Naive Bayes F1-macro: {nb_f1:.4f}")
print(f"Random Forest F1-macro: {rf_f1:.4f}")

# Select best model based on F1-macro
scores = {
    'Decision Tree': (dt, dt_f1), 
    'Naive Bayes': (nb, nb_f1), 
    'Random Forest': (rf, rf_f1)
}
best_name = max(scores, key=lambda k: scores[k][1])
best_model, best_score = scores[best_name]

# Save best model
with open('language_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nBest model: ({best_name}, F1={best_score:.4f}) saved as 'language_model.pkl'")
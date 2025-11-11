from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Naive Bayes classifier
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_test: Test features
    @param y_test: Test labels
    @return: Tuple of (model, accuracy, f1_score, predictions)
    """
    print("\n=== Training Naive Bayes ===")
    
    # Train Naive Bayes model
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    # Evaluate model
    nb_accuracy = nb.score(X_test, y_test) * 100
    y_pred_nb = nb.predict(X_test)
    nb_f1 = f1_score(y_test, y_pred_nb, average='macro')
    
    # Print results
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb))
    print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}%")
    print(f"Naive Bayes F1-macro: {nb_f1:.4f}")
    
    return nb, nb_accuracy, nb_f1, y_pred_nb
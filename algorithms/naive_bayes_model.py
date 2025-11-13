from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

def train_naive_bayes(X_train, y_train, X_test, y_test, X_val = None, y_val = None):
    """
    Train and evaluate Naive Bayes classifier
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_test: Test features
    @param y_test: Test labels
    @param X_val: Validation features (optional)
    @param y_val: Validation features (optional)
    @param alpha: Smoothing parameter for laplace smoothing
    @return: Tuple of (model, accuracy, f1_score, predictions)
    """
    print("\n=== Training Naive Bayes ===")

    # Grid for laplace smoothing
    grid = {'alpha': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.25, 1.50], # Not too high or not too low
            'fit_prior': [True, False],
            'class_prior': [None, [0.20, 0.75, 0.05]]} # in order: ENG FIL OTH

    # Initialize
    nb = MultinomialNB()
    grid_search = GridSearchCV(nb, grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    nb = grid_search.best_estimator_
    
    # Evaluate model
    nb_accuracy = nb.score(X_test, y_test) * 100
    y_pred_nb = nb.predict(X_test)
    nb_f1 = f1_score(y_test, y_pred_nb, average='macro')
    
    # Print results
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb))

    # print validation accuracy
    if X_val is not None and y_val is not None:
        nb_val_accuracy = nb.score(X_val, y_val) * 100
        print(f"Naive Bayes Validation Accuracy: {nb_val_accuracy:.2f}%")

    print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}%")
    print(f"Naive Bayes F1-macro: {nb_f1:.4f}")
    
    return nb, nb_accuracy, nb_f1, y_pred_nb
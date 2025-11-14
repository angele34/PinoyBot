from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Train and evaluate Decision Tree with Grid Search
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_test: Test features
    @param y_test: Test labels
    @param X_val: Validation features (optional)
    @param y_val: Validation labels (optional)
    @return: Tuple of (model, accuracy, f1_score, predictions)
    """
    print("\n=== Training Decision Tree with Grid Search ===")
    
    # Decision Tree with Hyperparameter Grid Search
    param_grid = {
        'criterion': ['gini', 'entropy'],          
        'max_depth': [3, 5, 7, 10, None],           
        'min_samples_split': [2, 5, 10],            
        'min_samples_leaf': [1, 2, 4],              
        'class_weight': [None, 'balanced'],         
        'ccp_alpha': [0.0, 0.001, 0.01],           
    }
    
    # Initialize and perform grid search
    dt_base = DecisionTreeClassifier(random_state=67)
    grid_search = GridSearchCV(dt_base, param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Decision Tree parameters: {grid_search.best_params_}")
    dt = grid_search.best_estimator_
    
    # Evaluate model
    dt_accuracy = dt.score(X_test, y_test) * 100
    y_pred = dt.predict(X_test)
    dt_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Print results
    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Decision Tree Test Accuracy: {dt_accuracy:.2f}%")
    print(f"Decision Tree F1-macro: {dt_f1:.4f}")
    
    return dt, dt_accuracy, dt_f1, y_pred
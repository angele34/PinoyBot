from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Train and evaluate Random Forest with Grid Search
    
    @param X_train: Training features
    @param y_train: Training labels
    @param X_test: Test features
    @param y_test: Test labels
    @param X_val: Validation features (optional)
    @param y_val: Validation labels (optional)
    @return: Tuple of (model, accuracy, f1_score, predictions)
    """
    print("\n=== Training Random Forest with Grid Search ===")
    
    rf_param_grid = {
        'n_estimators': [50],                    
        'criterion': ['entropy'],                
        'max_depth': [None],                     
        'min_samples_split': [10],               
        'min_samples_leaf': [1],              
        'max_features': ['sqrt'],                
        'class_weight': [None],      
    }
    
    # Initialize and perform grid search
    rf_base = RandomForestClassifier(random_state=67, n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring='f1_macro', verbose=1, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    print(f"\nBest Random Forest parameters: {rf_grid.best_params_}")
    rf = rf_grid.best_estimator_
    
    # Evaluate model
    rf_accuracy = rf.score(X_test, y_test) * 100
    y_pred_rf = rf.predict(X_test)
    rf_f1 = f1_score(y_test, y_pred_rf, average='macro')
    
    # Print results
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, digits=4))
    print(f"Random Forest Test Accuracy: {rf_accuracy:.2f}%")
    print(f"Random Forest F1-macro: {rf_f1:.4f}")
    
    return rf, rf_accuracy, rf_f1, y_pred_rf
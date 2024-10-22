from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Train and evaluate classifiers with parameter tuning
def train_and_evaluate(X_train, X_test, y_train, y_test):
    classifiers = {
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}),
        'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [5, 10]}),
        'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [50, 100]}),
        'SVM': (SVC(), {'kernel': ['linear', 'rbf'], 'C': [1, 10]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 10]}),
        'Neural Network': (MLPClassifier(max_iter=1000), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]})
    }

    performance_results = {}
    for clf_name, (clf, param_grid) in classifiers.items():
        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best estimator
        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        performance_results[clf_name] = {
            'Best Params': grid_search.best_params_,
            'Accuracy': accuracy,
            'Classification Report': report
        }
    
    return performance_results

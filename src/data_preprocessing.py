import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
# Load your dataset
def load_data(filepath):
    return pd.read_csv(filepath)

def impute_data(data, imputer):
    if imputer == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    if imputer == 'mean':
        imputer = SimpleImputer(strategy='mean')
    if imputer == 'median':
        imputer = SimpleImputer(strategy='median')
    if imputer == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    if imputer == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value=0)
    else:
        raise ValueError("Invalid imputer type. Please choose from 'knn', 'mean', 'median', 'most_frequent', or 'constant'.")
    # Apply the imputer to the dataset
    data_imputed = imputer.fit_transform(data)
    return data_imputed


# Preprocess your data
def preprocess_data(data):
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    # Apply label encoding to the 'RainToday' column
    data['RainToday'] = label_encoder.fit_transform(data['RainToday'])

    #Binarisation For Locations
    data = pd.get_dummies(data)
    # Feature selection methods

    # Example preprocessing steps
    X = data.drop('RainTomorrow', axis=1)  # Adjust for your target column
    y = data['RainTomorrow']
    return X, y






def feature_selection(X, y):
    # Apply feature selection techniques
    # Assuming X is your feature matrix and y is your target
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize your features (for Lasso and other models that need scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dictionary to store selected features and model performance
    feature_selection_results = {}

    # 1. Information Gain (Mutual Information)
    mi_scores = mutual_info_classif(X_train, y_train)
    mi_features = X.columns[mi_scores > 0.01]  # Choose a threshold for mutual information
    feature_selection_results['Mutual Information'] = mi_features

    # 2. Lasso (L1 Regularization)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train_scaled, y_train)
    lasso_features = X.columns[lasso.coef_ != 0]  # Features with non-zero coefficients
    feature_selection_results['Lasso'] = lasso_features

    # 3. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    rf_features = X.columns[rf_importances > 0.01]  # Select top features based on importance threshold
    feature_selection_results['Random Forest'] = rf_features

    # 4. Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)  # Adjust number of features
    rfe.fit(X_train, y_train)
    rfe_features = X.columns[rfe.support_]
    feature_selection_results['RFE'] = rfe_features

    # 5. Variance Threshold
    selector = VarianceThreshold(threshold=0.1)
    X_var_train = selector.fit_transform(X_train)
    var_features = X.columns[selector.get_support()]
    feature_selection_results['Variance Threshold'] = var_features
   

    return feature_selection_results  # Return selected features based on the methods
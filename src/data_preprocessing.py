import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer





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
    data['RainToday'] = label_encoder.fit_transform(data['RainToday'])

    #Binarisation For Locations
    data = pd.get_dummies(data)
    data = impute_data(data, imputer='knn')
    data = pd.DataFrame(data, columns=data.columns)
    # Split the data into features (X) and target (y)
    return data

def split_data(data, test_size=0.2, random_state=42):
    X = data.drop('RainTomorrow', axis=1)
    y = data['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer





def load_data(filepath):
    return pd.read_csv(filepath)


def impute_data(data):
    # Apply the KNN imputer to the dataset
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data)
    
    # Recreate DataFrame after imputation
    #data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
    
    return data_imputed


# Preprocess your data
def preprocess_data(data):
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    data['RainToday'] = label_encoder.fit_transform(data['RainToday'])

    #Binarisation For Locations
    data = pd.get_dummies(data)
    return data

def split_data(data, test_size=0.2, random_state=42):
    X = data.drop('RainTomorrow', axis=1)
    y = data['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


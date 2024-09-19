import pandas as pd

def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)#Load

    numeric_data = data.select_dtypes(include=['float64', 'int64'])#Only numerical columns
    categorical_data = data.select_dtypes(include=['object'])

    numeric_data.fillna(numeric_data.mean(), inplace=True)

    categorical_data.fillna('Unknown', inplace=True)#Handle missing values in categorical columns
    categorical_data_encoded = pd.get_dummies(categorical_data, drop_first=True)

    data_cleaned = pd.concat([numeric_data, categorical_data_encoded], axis=1)

    return data_cleaned

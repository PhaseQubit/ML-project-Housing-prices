import pandas as pd

def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)

    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    categorical_data = data.select_dtypes(include=['object'])

    numeric_data.fillna(numeric_data.mean(), inplace=True)
    categorical_data.fillna('Unknown', inplace=True)

    categorical_data_encoded = pd.get_dummies(categorical_data, drop_first=True)
    data_cleaned = pd.concat([numeric_data, categorical_data_encoded], axis=1)

    #Logging for data cleaning
    print(f"Data loaded and preprocessed. Shape of cleaned data: {data_cleaned.shape}")

    return data_cleaned

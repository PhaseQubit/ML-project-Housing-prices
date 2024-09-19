from data_preprocessing import load_and_preprocess_data
from modeling import train_and_evaluate_model
from visualization import plot_scatter_with_regression


def main():
    data = load_and_preprocess_data('AmesHousing.csv')#preprocessing function
    print(data.head())#Print first few rows of data to understand the data structure

    print("Columns in dataset:", data.columns)#Print all the columns
    plot_scatter_with_regression(data, 'Gr Liv Area', 'SalePrice')#Plot Gr Liv Area vs SalePrice

    mse, rmse, r2 = train_and_evaluate_model(data)#Train and evaluate the model
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'Root mean Squared Error: {rmse}')


if __name__ == "__main__":
    main()

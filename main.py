from data_preprocessing import load_and_preprocess_data
from modeling import train_and_evaluate_model
from visualization import plot_scatter_with_regression, plot_comparison


def main():
    data = load_and_preprocess_data('AmesHousing.csv')
    print(data.head())
    print("Columns in dataset:", data.columns)

    plot_scatter_with_regression(data, 'Gr Liv Area', 'SalePrice')

    mse_lr, rmse_lr, r2_lr, mse_rf, rmse_rf, r2_rf = train_and_evaluate_model(data)

    print(f'Linear Regression - MSE: {mse_lr}, RMSE: {rmse_lr}, R-squared: {r2_lr}')
    print(f'Random Forest - MSE: {mse_rf}, RMSE: {rmse_rf}, R-squared: {r2_rf}')

    plot_comparison(mse_lr, rmse_lr, mse_rf, rmse_rf)


if __name__ == "__main__":
    main()

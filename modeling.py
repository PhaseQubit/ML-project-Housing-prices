from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_model(data):
    X = data[['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Year Built']]
    y = data['SalePrice']

    #Split the data into training (60%), validation (20%), and testing (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    #Train and evaluate Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr_train = lr_model.predict(X_train)
    y_pred_lr_val = lr_model.predict(X_val)
    mse_lr_train = mean_squared_error(y_train, y_pred_lr_train)
    mse_lr_val = mean_squared_error(y_val, y_pred_lr_val)
    r2_lr = r2_score(y_val, y_pred_lr_val)
    rmse_lr = mse_lr_val ** 0.5

    #Train and evaluate Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf_train = rf_model.predict(X_train)
    y_pred_rf_val = rf_model.predict(X_val)
    mse_rf_train = mean_squared_error(y_train, y_pred_rf_train)
    mse_rf_val = mean_squared_error(y_val, y_pred_rf_val)
    r2_rf = r2_score(y_val, y_pred_rf_val)
    rmse_rf = mse_rf_val ** 0.5

    # Return MSE, RMSE, and R² for both models
    return mse_lr_train, rmse_lr, r2_lr, mse_rf_train, rmse_rf, r2_rf

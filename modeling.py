from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import mean_squared_error

def train_and_evaluate_model(data):
    X = data[['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Year Built']]  #features
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #predictions
    y_pred = model.predict(X_test)

    #Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5 #Root mean squared error
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, r2



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def plot_scatter_with_regression(data, feature, target):
    data = data.dropna(subset=[feature, target])

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=feature, y=target, data=data, color='green', label='Actual Data')

    X = data[[feature]]
    y = data[target]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.plot(data[feature], y_pred, color='red', linewidth=2, label='Regression Line')

    plt.title(f'{feature} vs {target} with Regression Line')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.legend()


    plt.show()

def plot_comparison(mse_lr_train, rmse_lr, mse_rf_train, rmse_rf):
    methods = ['Linear Regression', 'Random Forest']
    training_errors = [mse_lr_train, mse_rf_train]
    validation_errors = [rmse_lr, rmse_rf]

    bar_width = 0.35
    index = range(len(methods))

    fig, ax = plt.subplots(figsize=(10, 6))

    bar1 = ax.bar(index, training_errors, bar_width, label='Training Error')
    bar2 = ax.bar([i + bar_width for i in index], validation_errors, bar_width, label='Validation Error')

    ax.set_xlabel('Models')
    ax.set_ylabel('Mean Squared Error (Log Scale)')
    ax.set_title('Comparison of Training and Validation Errors')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(methods)
    ax.set_yscale('log')

    ax.legend()

    plt.show()

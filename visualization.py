import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def plot_scatter_with_regression(data, feature, target):
    data = data.dropna(subset=[feature, target])

    plt.figure(figsize=(10, 6))

    #Scatter plot of the feature vs. target
    sns.scatterplot(x=feature, y=target, data=data, color='green', label='Actual Data')

    #linear regression model
    X = data[[feature]]
    y = data[target]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    #Plot the regression line
    plt.plot(data[feature], y_pred, color='red', linewidth=2, label='Regression Line')

    plt.title(f'{feature} vs {target} with Regression Line')#titles and labels
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.legend()

    #Show
    plt.show()




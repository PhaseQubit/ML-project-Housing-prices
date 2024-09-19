This project aims to predict house prices based on various features of homes using the Ames Housing dataset. 
## Project Structure

The project is organized into several Python files:

- **main.py**: The main entry point of the project. It handles the data loading, model training, and evaluation.
- **data_preprocessing.py**: Contains the functions required for data cleaning, handling missing values, and preprocessing.
- **modeling.py**: Contains the code for training the model, splitting the dataset, and evaluating the model performance.
- **visualization.py**: Contains the code for generating visualizations such as scatter plots and regression lines.

## Dataset

The dataset used for this project is the **Ames Housing dataset**, which consists of 82 features describing 2,930 houses in Ames, Iowa. The target variable (`SalePrice`)
is the sale price of the houses, and the goal is to predict this value using various features of the houses.

- Dataset source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

### Features used:
- `Overall Qual`: Overall material and finish quality.
- `Gr Liv Area`: Above-ground living area in square feet.
- `Garage Cars`: Number of cars the garage can hold.
- `Total Bsmt SF`: Total basement square footage.
- `Year Built`: Year when the house was built.

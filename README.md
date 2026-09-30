# 🏠 House Price Prediction – Multiple Linear Regression

Predicting residential property prices in Tehran using Multiple Linear Regression, and measuring how much **location** improves the model.

## Overview
Two models are trained and compared:

| Model | Features | R² (test) |
|-------|----------|-----------|
| **Model 1** | Area, Room, Parking, Warehouse, Elevator | **0.54** |
| **Model 2** | Model 1 features + Address (one-hot encoded) | **0.76** |

Adding the neighbourhood feature raised R² from 0.54 to 0.76, showing that location is one of the strongest drivers of price.

## Dataset
`housePrice.csv` — 3,479 listings with the following columns:

`Area` · `Room` · `Parking` · `Warehouse` · `Elevator` · `Address` · `Price` · `Price(USD)`

Target variable: **Price_USD**

## Workflow
1. **Data cleaning** – removed missing values, converted `Area` to numeric, filtered outliers (Area < 600 m²)
2. **Feature engineering** – boolean features converted to integers; `Address` one-hot encoded (`drop_first=True`)
3. **EDA** – scatter plots of Area vs Price
4. **Modelling** – 80/20 train/test split, `LinearRegression` from scikit-learn
5. **Evaluation** – MSE and R² score on the test set

## Tech Stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Google Colab

## Getting Started
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
pip install pandas numpy scikit-learn matplotlib
jupyter notebook house_project.ipynb
```
> Update the CSV path in the notebook if you run it outside Google Colab.

## Project Structure
```
├── house_project.ipynb   # Main notebook
├── housePrice.csv        # Dataset
└── README.md
```

## Future Improvements
- Try regularised models (Ridge / Lasso) to handle the large number of address features
- Compare with tree-based models (Random Forest, XGBoost)
- Apply log-transformation to the price target
- Use cross-validation instead of a single random split

## Author
**Alireza Esmaeili**

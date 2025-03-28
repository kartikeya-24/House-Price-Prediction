# House Price Prediction

The "House Price Prediction" project focuses on predicting housing prices using machine learning techniques. By leveraging popular Python libraries such as NumPy, Pandas, Scikit-learn (sklearn), Matplotlib, Seaborn, and various regression models, this project provides an end-to-end solution for accurate price estimation.

## Project Overview

The "House Price Prediction" project aims to develop a model that can accurately predict housing prices based on various features. This prediction task is of great significance in real estate and finance, enabling informed decision-making for buyers, sellers, and investors. By employing machine learning algorithms and a curated dataset, this project provides a powerful tool for estimating house prices.

## Key Features

* *Data Collection and Processing:* The project utilizes the "California Housing" dataset, directly downloaded from the Scikit-learn library. This dataset includes features like house age, number of rooms, population, and median income. Pandas is used for efficient data processing and transformation to ensure suitability for analysis.
* *Data Visualization:* Matplotlib and Seaborn are employed to visualize the dataset, creating histograms, scatter plots, and correlation matrices. These visualizations aid in understanding feature relationships and identifying trends.
* *Train-Test Split:* The dataset is split into training and testing subsets using Scikit-learn's train_test_split to accurately evaluate the model's predictive capabilities on unseen data.
* *Regression Models:* This project explores and compares the performance of several regression models:
    * *Linear Regression:* A fundamental linear model for predicting continuous values.
    * *Decision Tree Regressor:* A tree-based model that makes predictions by learning simple decision rules inferred from the data features.
    * *Support Vector Regression (SVR):* A powerful model that uses support vector machines for regression tasks.
    * *Artificial Neural Network (MLPRegressor):* A neural network model for complex non-linear regression.
* *Model Evaluation:* The performance of each regression model is assessed using metrics such as:
    * *R-squared (R2) score:* Measures the proportion of variance in the target variable explained by the model.
    * *Mean Absolute Error (MAE):* Quantifies the average magnitude of errors in the predictions.
    * *Mean Squared Error (MSE):* Measures the average squared difference between predicted and actual values.
    * *Root Mean Squared Error (RMSE):* The square root of MSE, providing an interpretable error measure.
    * Scatter plots are also used to visualize predicted vs. actual prices.
* *Feature Scaling:* StandardScaler is used to standardize the features before training the models, improving the performance of algorithms sensitive to feature scaling, such as SVR and ANN.

## Project Modules

1.  *Data Preprocessing:*
    * Handles missing values using median imputation.
    * Splits the data into features (X) and target (Y).
2.  *Feature Scaling:*
    * Standardizes the features using StandardScaler to ensure consistent scaling.
3.  *Model Training and Evaluation:*
    * Trains and evaluates multiple regression models.
    * Calculates and displays evaluation metrics for each model.
    * Creates scatter plots of Actual vs Predicted values for Linear Regression, SVR, and ANN.
    * Creates a visualization of the Decision Tree model.
    * Creates feature importance graphs for Random Forest, Gradient Boosting, and XGBoost models.
4.  *Model Comparison:*
    * Compares the performance of different models using a bar plot of R2 scores.

## Libraries Used

* NumPy
* Pandas
* Scikit-learn (sklearn)
* Matplotlib
* Seaborn

This project provides a comprehensive analysis of house price prediction, demonstrating the application of various machine learning techniques and providing valuable insights for real-world scenarios.

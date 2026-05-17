# Ames Housing Price Prediction & Advanced Regression

An end-to-end machine learning project utilizing the Ames Housing Dataset to predict residential property prices. This project focuses on rigorous data preprocessing, preventing data leakage, custom feature engineering, and optimizing regularized linear models alongside ensemble tree methods.

## 🚀 Key Features & Workflow
* **Data Preprocessing & Encoding:** Handled missing data logically. Employed manual Ordinal Encoding for ranked quality metrics and Scikit-Learn's `OneHotEncoder` for nominal features inside a strict train/test boundary to eliminate data leakage.
* **Feature Engineering:** Boosted model efficiency by engineering spatial and temporal metrics: `Total_Living_SF`, `Total_Baths`, and `House_Age_At_Sale`.
* **Hyperparameter Tuning:** Utilized `GridSearchCV` with 5-fold cross-validation to search optimal regularization strengths ($\alpha$) for Lasso, Ridge, and ElasticNet models.

## 📊 Final Model Performance

We evaluated five distinct algorithms using Root Mean Squared Error (RMSE). Regularization significantly improved standard linear performance, with Ridge Regression narrowing the gap against an ensemble tree model.

| Regression Model | Test RMSE | Performance Ranking |
| :--- | :---: | :---: |
| **Random Forest (Engineered)** | **25,110.50** | 🏆 1st Place |
| **Ridge Regression ($\alpha=10$)** | **25,145.30** | 🥈 2nd Place |
| **ElasticNet Regression** | **25,367.01** | 3rd Place |
| **Lasso Regression ($\alpha=50$)** | **25,494.50** | 4th Place |
| **Standard Linear Regression** | **25,572.37** | 5th Place |

### Model Insights & Evaluation Visualizations

![Model Comparison](https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/ames-housing-prediction/main/images/model_comparison.png)

*Figure 1: Comparison of model prediction error (RMSE) demonstrating how introducing mathematical penalties (Ridge/Lasso) stabilized standard linear regression.*

![Predictions vs Actual](https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/ames-housing-prediction/main/images/predictions_vs_actual.png)

*Figure 2: Random Forest predicted values plotted against actual test sales prices, indicating a strong $R^2$ accuracy of ~0.92.*

## 🛠️ How to Run This Project

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/ames-housing-prediction.git](https://github.com/YOUR_GITHUB_USERNAME/ames-housing-prediction.git)

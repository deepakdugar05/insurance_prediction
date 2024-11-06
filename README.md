
# Insurance Cost Prediction

This project focuses on building a predictive model to estimate insurance charges for customers based on specific factors such as age, gender, BMI, and other personal details. This model can assist insurance companies in determining fair premiums and understanding factors that influence costs.

## Project Overview

The project uses machine learning algorithms to predict insurance charges by analyzing patterns in customer data. Various regression models are explored to optimize predictions and select the most accurate model.

### Dataset

The dataset, `insurance_data.csv`, includes features relevant to insurance costs, such as:

- **Age**: Age of the insured.
- **Gender**: Gender of the insured.
- **BMI**: Body Mass Index, a measure of body fat.
- **Children**: Number of children/dependents covered by insurance.
- **Smoker**: Smoking status of the insured.
- **Region**: Geographic region of the insured.
- **Charges**: Insurance cost, the target variable.

### Objective

The objective is to build a model to accurately predict the `charges` variable based on the other features in the dataset.

## Steps and Methodology

1. **Data Preprocessing**:
   - Loaded data and performed cleaning and preprocessing.
   - Scaled features using `StandardScaler` for normalization.
   - Applied transformations like `PowerTransformer` to reduce skewness in data.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized data distributions, correlations, and identified significant features impacting insurance charges.
   - Explored trends such as the relationship between `BMI` and `charges` or the effect of smoking status on insurance cost.

3. **Model Building**:
   - Split the data into training and testing sets.
   - Evaluated several regression models:
     - **Linear Regression**
     - **Ridge Regression**
     - **Lasso Regression**
     - **ElasticNet**
     - **Decision Tree Regressor**
     - **Random Forest Regressor**
     - **K-Nearest Neighbors**
     - **Support Vector Regressor**
   - Used `GridSearchCV` for hyperparameter tuning to optimize model performance.

4. **Model Evaluation**:
   - Compared models using evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
   - Performed cross-validation to ensure generalizability.

## Key Findings

- Certain factors, such as `smoking status` and `BMI`, have a significant impact on insurance charges.
- Among the models tested, ensemble methods like **Random Forest Regressor** showed high accuracy in predicting insurance costs.

## Conclusion

This project successfully builds a predictive model for insurance costs, leveraging regression techniques and data processing. Such a model provides valuable insights for insurance providers to better understand cost determinants and set appropriate premiums.

## Installation

To run this project, install the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository.
2. Load the dataset `insurance_data.csv` into the project directory.
3. Run the Jupyter Notebook `insurance_prediction.ipynb` to see the analysis and model training steps.

## License

This project is licensed under the MIT License.


# Review Prediction Using Machine Learning

This project demonstrates the prediction of `review/overall` ratings for beer reviews using machine learning models. The data comes from a beer review dataset, and the project applies various preprocessing techniques, feature engineering, and model training strategies to achieve accurate predictions.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Key Results](#key-results)
6. [Installation and Usage](#installation-and-usage)
7. [Future Enhancements](#future-enhancements)

---

## Dataset Overview

The dataset consists of 37,500 beer reviews with 18 features:
- **Numerical Features**: `beer/ABV`, `review/appearance`, `review/aroma`, etc.
- **Text Features**: `beer/name`, `review/text`.
- **Categorical Features**: `beer/style`.
- **Target Variable**: `review/overall`.

### Cleaning Steps:
- Removed rows where `review/appearance` or `review/overall` were less than 1.
- Dropped unnecessary columns with significant missing data.
- Handled outliers in numerical data by correlation analysis.

---

## Data Preprocessing

### Steps:
1. **Handling Missing Values**:
   - Dropped columns: `user/ageInSeconds`, `user/birthdayUnix`, `user/birthdayRaw`, `user/gender`.
   - Removed rows with missing values in `review/text` or `user/profileName`.

2. **Standardization & Encoding**:
   - Standardized numerical features using `StandardScaler`.
   - Encoded categorical features using `OneHotEncoder`.

3. **Text Preprocessing**:
   - Concatenated `beer/name`, `beer/style`, and `review/text` into a unified text column.
   - Removed stop words and irrelevant terms using `TfidfVectorizer`.

---

## Feature Engineering

### Key Features:
1. **Numerical**: `review/appearance`, `review/aroma`, `review/palate`, `review/taste`, `review/timeUnix`.
2. **Categorical**: `beer/style`.
3. **Text**: Processed `review/text` and extracted TF-IDF features.

---

## Model Training and Evaluation

### Test Scenarios:
1. **Test 1**: Numerical and categorical features only.
2. **Test 2**: Numerical, categorical, and text features combined.
3. **Test 3**: Text features only.

### Models Used:
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Support Vector Regressor**

### Results:
| Test Scenario | Model                  | R² Score | Mean Squared Error |
|---------------|------------------------|----------|--------------------|
| Test 1        | XGBoost Regressor      | 0.689    | 0.153              |
| Test 2        | XGBoost Regressor      | 0.698    | 0.149              |
| Test 3        | Random Forest/XGBoost  | ~0.308   | 0.341              |

- **Conclusion**: Test 1, which utilizes only numerical and categorical features, achieves a slightly lower R² score compared to Test 2. However, it offers faster training time due to the absence of text preprocessing and handling, making it a more efficient choice for scenarios where computational resources or time are limited

---

## Key Results

1. **Best Performing Model**: 
   - **Test 1**: Utilizes only numerical and categorical features, offering the fastest training time. It is an efficient choice for scenarios where computational resources or time are limited, despite slightly lower predictive accuracy compared to Test 2.

2. **Best Predictive Model**: 
   - **Test 2**: Combines numerical, categorical, and text features, achieving the highest accuracy with an R² score of **0.698** and a Mean Squared Error of **0.149**. This model is the ideal choice for scenarios where accuracy is prioritized over training time.

3. **Insights**:
   - Text data alone (Test 3) is insufficient for accurate predictions, as it yields an R² score of only ~0.3.
   - Numerical and categorical features (Test 1) demonstrate strong predictive power while being computationally efficient.
   - Combining text with numerical and categorical features (Test 2) enhances overall performance, delivering the best predictive accuracy.
---
## Author Information

- **Name**: Van Tai Huynh
- **Specialization**: Computer Programming and Analytics at Seneca Polytechnic
- **Expertise**: Data Science, Machine Learning, and Parallel Programming
- **Contact**: [hvtai.it@gmail.com] | [LinkedIn https://www.linkedin.com/in/van-tai-huynh/ ]
- 


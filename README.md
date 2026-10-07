# Disease Prediction

## Overview
This project uses machine learning to predict the presence of heart disease from structured medical data. We built an end-to-end classification pipeline that handles missing values, scales numerical features, encodes categorical variables, and evaluates different models. 

**Note**: This is an educational machine-learning project and NOT a medical diagnostic system.

## Dataset
- **Source**: UCI Machine Learning Repository (Processed Cleveland dataset)
- **Features**: 13 clinical attributes (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression, slope of the peak exercise ST segment, number of major vessels, and thalassemia).
- **Target Variable**: Presence of heart disease (Binary classification: 0 = No Disease, 1 = Disease).

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Preprocessing
- **Missing Values**: Imputed using median for numerical features and most frequent value for categorical features.
- **Numerical Scaling**: Standardized using `StandardScaler`.
- **Categorical Encoding**: One-hot encoded using `OneHotEncoder`.
- **Data Leakage Prevention**: All preprocessing steps were placed inside a `ColumnTransformer` and embedded within a scikit-learn `Pipeline`, ensuring they are only fitted on the training set after the train/test split.
- **Train/Test Split**: Stratified 80/20 split.

## Models
We trained and evaluated the following classification models:
- **Logistic Regression**
- **Random Forest**

## Evaluation
Both models were evaluated on the test set. For this medical context, **Recall** was heavily emphasized because failing to identify a patient with a disease (a false negative) carries severe risks.

The best-performing model was **Logistic Regression**, which achieved the following metrics on the test set:
- **Accuracy**: 0.8852
- **Precision**: 0.8387
- **Recall**: 0.9286
- **F1 Score**: 0.8814
- **ROC-AUC**: 0.9665

## Results
Logistic Regression outperformed the Random Forest model in Recall and had an excellent ROC-AUC score, indicating it is robust and highly sensitive to positive cases. The `results/` folder contains generated visualizations:
- **`model_comparison.png`**: A bar chart comparing metrics across the evaluated models.
- **`roc_curve.png`**: ROC curves for both models.
- **`confusion_matrix.png`**: The confusion matrix of the Logistic Regression model, showing the breakdown of true vs predicted labels.

## Medical Disclaimer
This project is for educational purposes only and must not be used as a medical diagnostic or treatment system. Real clinical systems require clinical validation, appropriate patient data, regulatory review, calibration, fairness evaluation, and expert oversight.

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook Disease_Prediction.ipynb
   ```
3. Run all cells to reproduce the results, visualizations, and the exported model.

## Conclusion
The project successfully demonstrates a machine learning pipeline capable of predicting heart disease with high recall and overall accuracy. The resulting Logistic Regression model pipeline handles its own preprocessing and is saved as `models/best_disease_model.pkl` for easy loading and inference.


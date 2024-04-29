# NextDayRainPrediction
- data access via kaggle: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?select=weatherAUS.csv (April 2024)

## EDA
Currently the EDA part consists of histogram of numerical features and correlation heatmap. More EDA is to be done.

## Preprocessing - Ain
This data is preprocessed by several steps:
1. Data Cleaning
    - remove null rows
    - new column `Month` is introduced by extracting the month from column `Date`, whereas the column `Date` is dropped.
2. Handling Outlier
    - Outliers are handled through z-score, where any z-score that exceeds the set threshold (`outlier_threshold = 3`) is filtered out.
3. One-hot encoding
    -  One-hot encoding is applied to the data to handle categorical data for modeling steps.
4. Data imbalance
    - Data imbalanced was observed in the original dataset, therefore two steps are applied:
        - SMOTE
        - Random Undersampling

## Feature Selection - Ain
1. For numerical features, anova f-test was performed to observe the f-score of the datasets. This was applied to all three original preprocessed_df, preprocessed_smote_data, and RUS_df.
    - Each dataset had different ranking of f-scores where the top 5 features are:
        - `Preprocessed_df: Temp9am, MinTemp, Rainfall, WindSpeed3pm, MaxTemp`
        - `smote_df: Rainfall, WindGustSpeed, WindSpeed3pm, Temp9am, MinTemp`
        - `RUS_df: Rainfall, MaxTemp, Temp9am, MinTemp, Temp3pm`
2. For categorical feature, chi2 test was done to observe the chi2 score. For all three datasets, the `RainToday_Yes` feature had the most outstanding score, whereas other categorical features do not show significance in predicting RainTomorrow.

## Exported data 
Since only `RainToday_Yes` show high significance to `RainToday_Yes`, the exported data will include all numerical features and only `RainToday_Yes` from the categorical feature. All numerical features are added for flexibility of feature selection during modeling, and dropping other categorical features from the one-hot encoded dataset will reduce less processing time on the SAS EnterpriseMiner software.

Exported data are as follows:
1. `preprocessed_data.csv`
2. `preprocessed_data_smote.csv`
3. `preprocessed_data_RUS.csv`

The target variable shall be RainTomorrow_Yes. 
No splitting is done in this step as splitting will be done during modeling in the SAS EnterpriseMiner.

Last Updated: 29 April 2024

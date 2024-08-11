# %% [markdown]
# # Predicting Next Day Rainfall

# %% [markdown]
# # 1.0   Sample

# %%
import pandas as pd

df = pd.read_csv("weatherAUS.csv")
df.head()


# %%
print("Number of rows:", len(df))

# %% [markdown]
# ### 1.1 Data Understanding
# 
# ![Alt text](image.png "Data Description")
# 
# *Hassan et al (2023)*
# 
# 

# %%
# Display the data types of each variable
print(df.dtypes)

# %% [markdown]
# ### 1.2 Data Cleaning

# %%
# Count the number of rows with NaN or missing values
null_rows = df.isnull().sum()
print(null_rows)

# %%
# Check for duplicated rows
duplicated_rows = df[df.duplicated()]

# Print the duplicated rows if any
if not duplicated_rows.empty:
    print("Duplicated Rows:")
    print(duplicated_rows)
else:
    print("No duplicated rows found.")

# %%
weather_df = df.dropna()

# Calculate the percentage of rows in weather_df compared to total rows in df
percentage = (len(weather_df) / len(df)) * 100

# Print the percentage with one decimal place
print(f"Percentage of rows retained: {percentage:.1f}%")

# Print number of rows retained
print("Number of rows retained: ", len(weather_df))

# %%
# Convert 'Date' column to datetime datatype
weather_df['Date'] = pd.to_datetime(weather_df['Date'])

# Extract the month and create a new column 'Month'
weather_df['Month'] = weather_df['Date'].dt.month

# Drop the original 'Date' column
weather_df.drop('Date', axis=1, inplace=True)

# Move the 'Month' column as the first column
weather_df = weather_df[['Month'] + [col for col in weather_df.columns if col != 'Month']]


# %%
# Update non-numerical data tyopes
columns_to_convert = ['Month','WindGustDir', 'WindDir9am', 'WindDir3pm', 'Cloud9am', 'Cloud3pm' ]
weather_df[columns_to_convert] = weather_df[columns_to_convert].astype('category')

# %%
# Replacing Yes and No value to 1 and 0 respectively to binary variables

weather_df['RainToday'] = weather_df['RainToday'].replace({'Yes': 1, 'No': 0})
weather_df['RainTomorrow'] = weather_df['RainTomorrow'].replace({'Yes': 1, 'No': 0})


# %%
weather_df.head()

# %%
weather_df.tail()

# %%
# Display the data types of each variable
print(weather_df.dtypes)

# %% [markdown]
# ## 2.0 Explore
# 
# 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Select only numeric columns for correlation calculation
numeric_columns = weather_df.select_dtypes(include=['float64'])

# %%
# Set up the subplot grid
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through each numeric column and plot histogram
for i, col in enumerate(numeric_columns):
    sns.histplot(weather_df[col], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# # Check class balance for RainTomorrow
# class_counts = weather_df['RainTomorrow'].value_counts()
# plt.figure(figsize=(6, 4))
# sns.barplot(x=class_counts.index, y=class_counts.values)
# plt.title('Class Balance for RainTomorrow')
# plt.xlabel('RainTomorrow (0: No, 1: Yes)')
# plt.ylabel('Frequency')
# plt.show()

# %%
# Calculate the correlation matrix for numeric columns
corr_matrix = numeric_columns.corr()

# Generate a heatmap using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# %%
# Select relevant columns for the pairplot
columns_of_interest = ['RainTomorrow', 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'Evaporation', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']
relevant_data = weather_df[columns_of_interest]


# Generate pairplot using seaborn
sns.pairplot(relevant_data, hue='RainTomorrow', markers=['o', 's'])
plt.suptitle('Ensemble of Response vs. Features Scatter Plots', y=1.02)
plt.show()

# %% [markdown]
# ## 3.0 Modify
# 
# ### 3.1 One Hot Encoding

# %%
# List of categorical columns
categorical_columns = ['Month', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Cloud9am', 'Cloud3pm']

# Apply one-hot encoding to categorical columns
weather_encoded = pd.get_dummies(weather_df, columns=categorical_columns)

# Print the first few rows of the encoded DataFrame
print(weather_encoded.head())

# %%
# Display the data types of each variable
print(weather_encoded.dtypes)

# %% [markdown]
# ### 3.2 Handling Data Outliers

# %%
# Plot boxplot before outlier removal
plt.figure(figsize=(10, 6))
sns.boxplot(data=numeric_columns, orient='v', palette='Set2')
plt.title('Boxplot Before Outlier Removal')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
from scipy.stats import zscore

# %%
# Compute z-scores for numerical columns
z_scores = zscore(weather_df.select_dtypes(include=['float64']))

# Define threshold for identifying outliers 
outlier_threshold = 3

# Filter rows where any z-score exceeds the threshold
outliers_removed = weather_df[(abs(z_scores) < outlier_threshold).all(axis=1)]

# Print number of outliers removed
print(f"Number of outliers removed: {len(weather_df) - len(outliers_removed)}")

# %%
# Plot boxplot after outlier removal
plt.figure(figsize=(10, 6))
sns.boxplot(data=outliers_removed.select_dtypes(include=['float64', 'int64']), orient='v', palette='Set2')
plt.title('Boxplot After Outlier Removal')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Feature Importance
# 
# #### 3.2.1 Point-Biserial Correlation

# %%
from scipy.stats import pearsonr, pointbiserialr, ttest_ind

# %%
# Calculate Pearson correlation for each numerical column with RainTomorrow
pearson_correlations = {}
for col in numeric_columns:
    correlation, p_value = pearsonr(weather_df[col], weather_df['RainTomorrow'])
    pearson_correlations[col] = {'Correlation': correlation, 'P-Value': p_value}

# Calculate Point-Biserial correlation for binary target variable
point_biserial_results = {}
for col in numeric_columns:
    point_biserial_correlation, p_value = pointbiserialr(weather_df['RainTomorrow'], weather_df[col])
    point_biserial_results[col] = {'Point-Biserial Correlation': point_biserial_correlation, 'P-Value': p_value}

# Perform t-test for significance of correlation
t_test_results = {}
for col in numeric_columns:
    t_stat, t_p_value = ttest_ind(weather_df.loc[weather_df['RainTomorrow'] == 1, col], weather_df.loc[weather_df['RainTomorrow'] == 0, col])
    t_test_results[col] = {'T-Statistic': t_stat, 'P-Value': t_p_value}

# Convert results to DataFrames for better visualization
pearson_df = pd.DataFrame.from_dict(pearson_correlations, orient='index')
point_biserial_df = pd.DataFrame.from_dict(point_biserial_results, orient='index')
t_test_df = pd.DataFrame.from_dict(t_test_results, orient='index')

# Print results
print("\nPoint-Biserial Correlation and Significance:")
print(point_biserial_df)

print("\nT-Test Results:")
print(t_test_df)

# %%
# Set the figure size
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(point_biserial_df, annot=True, cmap='coolwarm', fmt=".3f")

# Set the title and labels
plt.title('Point-Biserial Correlation with RainTomorrow')
plt.xlabel('Features')
plt.ylabel('Point-Biserial Correlation')

# Display the plot
plt.show()

# %% [markdown]
# #### 3.2.2 Chi-Square Test

# %%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# %% [markdown]
# ### 3.3 Data Imbalance
# 

# %%
# Count the occurrences of each class in the target variable
class_counts = weather_df['RainTomorrow'].value_counts()

# Calculate the class imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]

# Print the class counts and imbalance ratio
print("Class Counts:")
print(class_counts)
print("\nImbalance Ratio (0s to 1s):", imbalance_ratio)

# Plotting the class distribution
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution for RainTomorrow')
plt.xlabel('RainTomorrow (0: No Rain, 1: Rain)')
plt.ylabel('Counts')
plt.xticks(rotation=0)
plt.show()



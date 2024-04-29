import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

# Load the dataset
terrorism_df = pd.read_csv(r"C:\Users\shric\Desktop\Dai\articles\CAPSTONE\Predictive_Analysis_GTDB\data\globalterrorismdb_0522dist.csv")

# Clean the dataset
columns_of_interest = ['iyear', 'country_txt', 'region_txt', 'latitude', 'longitude', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt', 'success', 'nkill', 'nwound', 'gname']
cleaned_terrorism_df = terrorism_df[columns_of_interest]

# Filter data for the specified time period
cleaned_terrorism_df = cleaned_terrorism_df[(cleaned_terrorism_df['iyear'] >= 2000) & (cleaned_terrorism_df['iyear'] <= 2023)]
cleaned_terrorism_df.info()

# Visualize top countries by number of attacks
top_countries = cleaned_terrorism_df['country_txt'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number Of Attacks')
plt.title('Top Countries by Number of Attacks')
plt.xticks(rotation=70)
plt.show()

# Count the number of each type of attack
attack_counts = cleaned_terrorism_df['attacktype1_txt'].value_counts().head(5)

# Set index as attack count
filtered_df = cleaned_terrorism_df[cleaned_terrorism_df['attacktype1_txt'].isin(attack_counts.index)]

# Count number of attacks per country
country_attack_counts = filtered_df.groupby(['country_txt', 'attacktype1_txt']).size().unstack(fill_value=0)

# Set index for country attacks
country_attack_counts = country_attack_counts.loc[top_countries.index].reset_index().melt(id_vars='country_txt', var_name='Type of Attack', value_name='Number of Attacks')

# Top countries by Number of Attacks and Type
plt.figure(figsize=(12, 8))
sns.barplot(data=country_attack_counts, x='country_txt', y='Number of Attacks', hue='Type of Attack', palette='muted')
plt.xlabel('Country')
plt.ylabel('Number of Attacks')
plt.title('Top Countries by Number of Attacks and Top Types of Attack')
plt.legend(title='Type of Attack', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()

# Get the top 10 target types by the number of attacks
top_targets = filtered_df['targtype1_txt'].value_counts().head(10).index

# Filter the DataFrame to include only the top 10 target types
filtered_df_top_targets = filtered_df[filtered_df['targtype1_txt'].isin(top_targets)]

# Number of attacks over the years for the top 10 target types
plt.figure(figsize=(12, 8))
sns.countplot(data=filtered_df_top_targets, x='targtype1_txt', hue='region_txt', palette='tab20')
plt.xlabel('Target Type')
plt.ylabel('Number of Attacks')
plt.title('Number of Attacks by Top 10 Target Types and Region')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Group the data by year and count the number of attacks in each year
attacks_over_time = filtered_df['iyear'].value_counts().sort_index()

# Plotting the number of attacks over time using Seaborn
plt.figure(figsize=(12, 6))
sns.scatterplot(x=attacks_over_time.index, y=attacks_over_time.values)
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.title('Number of Attacks Over Time')
plt.annotate('Peak 2014', xy=(2014, 16000), xytext=(2010, 15900), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', linewidth=3))
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# Group data by country and sum the number of fatalities
fatalities_by_country = filtered_df.groupby('country_txt')['nkill'].sum().reset_index()

# Sort the DataFrame by the number of fatalities in descending order
fatalities_by_country = fatalities_by_country.sort_values(by='nkill', ascending=False)

# Visualize the number of fatalities by country
plt.figure(figsize=(12, 8))
sns.barplot(data=fatalities_by_country.head(10), x='nkill', y='country_txt', palette='viridis')
plt.xlabel('Number of Fatalities')
plt.ylabel('Country')
plt.title('Number of Fatalities by Country (Top 15)')
plt.tight_layout()
plt.show()

# Prepare data for Geospatial Analysis
geo_spatial_data = filtered_df[['latitude', 'longitude', 'nkill', 'attacktype1_txt']]

# Filter out attacks with missing latitude or longitude values
geo_spatial_data = geo_spatial_data.dropna(subset=['latitude', 'longitude'])

# Create a map centered around the mean latitude and longitude
attack_map = folium.Map(location=[geo_spatial_data['latitude'].mean(), geo_spatial_data['longitude'].mean()], zoom_start=2)

# Add a marker for each attack
for index, row in geo_spatial_data.iterrows():
    folium.Marker(location=[row['latitude'], row['longitude']],
                  popup=row['attacktype1_txt'] + ' - ' + str(row['nkill']) + ' fatalities',
                  icon=folium.Icon(color='red')).add_to(attack_map)

# Display the map
attack_map

# Filter the DataFrame to include only the top ten countries
top_countries_df = cleaned_terrorism_df[cleaned_terrorism_df['country_txt'].isin(top_countries.index)]

# One-hot encode the categorical columns
country_dummies = pd.get_dummies(top_countries_df[['country_txt', 'region_txt', 'attacktype1_txt', 'weaptype1_txt', 'targtype1_txt', 'gname']])

# Concatenate one-hot encoded columns with the original DataFrame
country_dummies = pd.concat([top_countries_df, country_dummies], axis=1)

# Drop the original categorical columns
country_dummies.drop(columns=['country_txt', 'region_txt', 'attacktype1_txt', 'weaptype1_txt', 'targtype1_txt', 'gname'], inplace=True)

# Now country_dummies contains the filtered data with one-hot encoded categorical columns and the 'summary' column dropped
country_dummies.head(1)

# selected_columns = ['iyear', 'country_txt', 'region_txt', 'latitude', 'longitude', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt', 'success', 'nkill', 'nwound', 'gname']

# # Create a new DataFrame with selected columns
# log_model_columns = country_dummies[selected_columns]

# log_model_columns
country_dummies.dropna()
country_dummies.info()
country_dummies_corr = country_dummies.corr()
print(country_dummies_corr)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the region_txt column
filtered_df['region'] = label_encoder.fit_transform(filtered_df['region_txt'])

# Print the mapping of encoded values to region_txt categories
print("Encoded values for region_txt:")
for i, region in enumerate(label_encoder.classes_):
    print(f"{region}: {i}")

filtered_df.head(1)
contingency_table = pd.crosstab(filtered_df['attacktype1_txt'], filtered_df['region'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Print test statistics
print("Chi-square statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected frequencies:")
print(expected)

# Plot heatmap of observed frequencies
plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Observed Frequencies of Attacks by Type and Region")
plt.xlabel("Region Code")
plt.ylabel("Type of Attack")
plt.show()

# Combine X and y into a single DataFrame
combined_df = country_dummies.dropna(subset=['success'])

# Split data into features (X) and target variable (y)
X = combined_df.drop(columns=['success'])
y = combined_df['success']

# Now check if the lengths of X and y are the same
print("Length of X before preprocessing:", len(X))
print("Length of y before preprocessing:", len(y))

# Drop rows with missing values from both X and y
combined_df_cleaned = combined_df.dropna(how='any')
X = combined_df_cleaned.drop(columns=['success'])
y = combined_df_cleaned['success']

# Align indices of X and y
X = X.loc[y.index]

# Now check if the lengths of X and y are the same after preprocessing
print("Length of X after preprocessing:", len(X))
print("Length of y after preprocessing:", len(y))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit logistic regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report1)
print("ROC AUC Score:", roc_auc)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, logistic_model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:,1])  # Corrected here

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

attack_country_dummies = pd.get_dummies(top_countries_df[['country_txt', 'region_txt', 'attacktype1_txt', 'weaptype1_txt', 'targtype1_txt', 'gname']])

# Identify the attack type columns
attack_type_columns = [col for col in attack_country_dummies.columns if 'attacktype1_txt_' in col]

# Extract the attack type columns along with other necessary features
features_and_target = attack_country_dummies[attack_type_columns]
attack_country_dummies

# Define other features excluding attack type columns
other_features = [col for col in country_dummies.columns if col not in attack_type_columns]

# Identify the attack type columns
attack_type_columns = [col for col in country_dummies.columns if 'attacktype1_txt_' in col]

# Extract the attack type columns along with other necessary features
features_and_target = country_dummies[attack_type_columns + other_features]

# Drop rows with missing values from both X_multiclass and y_multiclass
X_multiclass = X_multiclass.dropna()
y_multiclass = y_multiclass.loc[X_multiclass.index]

# Convert one-hot encoded target variable to class labels
y_multiclass_labels = np.argmax(y_multiclass.values, axis=1)

# Split data into training and testing sets
X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(X_multiclass, y_multiclass_labels, test_size=0.3, random_state=42)

# Fit logistic regression model
logistic_model_multiclass = LogisticRegression(max_iter=1000)
logistic_model_multiclass.fit(X_train_multiclass, y_train_multiclass)

# Predict on the test set
y_pred_multiclass = logistic_model_multiclass.predict(X_test_multiclass)

# Evaluate model performance
accuracy_multiclass = accuracy_score(y_test_multiclass, y_pred_multiclass)
classification_report_multiclass = classification_report(y_test_multiclass, y_pred_multiclass)

print("Accuracy:", accuracy_multiclass)
print("Classification Report:\n", classification_report_multiclass)

# Confusion Matrix
conf_matrix_multiclass = confusion_matrix(y_test_multiclass, y_pred_multiclass)

# Plot Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_multiclass, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix - Multiclass Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

columns_of_interest = ['iyear', 'country_txt', 'region_txt', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt', 'nkill', 'nwound']
cleaned_df = terrorism_df[columns_of_interest]

# Filter data for the specified time period
cleaned_df = cleaned_df[(cleaned_df['iyear'] >= 2000) & (cleaned_df['iyear'] <= 2023)]

# Drop rows with missing values
cleaned_df.dropna(inplace=True)

# One-hot encode categorical features
encoded_df = pd.get_dummies(cleaned_df, columns=['country_txt', 'region_txt', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt'])

# Separate features (X) and target variable (y)
X = encoded_df.drop(columns=['nkill', 'nwound'])  # Features
y = encoded_df[['nkill', 'nwound']]  # Target variable (number of fatalities and injuries)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forests Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
plt.figure(figsize=(10, 6))
plt.scatter(y_test['nkill'], y_pred[:, 0], color='blue', label='Predicted')
plt.plot([0, max(y_test['nkill'])], [0, max(y_test['nkill'])], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual Number of Fatalities')
plt.ylabel('Predicted Number of Fatalities')
plt.title('Actual vs. Predicted Number of Fatalities')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test['nwound'], y_pred[:, 1], color='green', label='Predicted')
plt.plot([0, max(y_test['nwound'])], [0, max(y_test['nwound'])], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual Number of Injuries')
plt.ylabel('Predicted Number of Injuries')
plt.title('Actual vs. Predicted Number of Injuries')
plt.legend()
plt.grid(True)
plt.show()

# Filter out rows with missing latitude or longitude values
geo_data = cleaned_terrorism_df.dropna(subset=['latitude', 'longitude'])

# Group data by year and count the number of attacks in each location
geo_data_grouped = geo_data.groupby(['iyear', 'latitude', 'longitude']).size().reset_index(name='attack_count')

# Create a base map centered around the mean latitude and longitude
base_map = folium.Map(location=[geo_data_grouped['latitude'].mean(), geo_data_grouped['longitude'].mean()], zoom_start=2)

# Create a list of lists containing location data and attack count for each year
heat_data = [[[row['latitude'], row['longitude'], row['attack_count']] for index, row in geo_data_grouped[geo_data_grouped['iyear'] == year].iterrows()] for year in sorted(geo_data_grouped['iyear'].unique())]

# Create HeatMapWithTime layer
HeatMapWithTime(heat_data, radius=15).add_to(base_map)

# Display the map
base_map.save("terrorist_attacks_heatmap_with_time.html")

# Define numerical values for each target type category
target_type_values = {
    'Business': 1,
    'Police': 2,
    'Private Citizens & Property': 3,
    'Utilities': 4,
    'Military': 5,
    'Violent Political Party': 6,
    'Government (General)': 7,
    'Transportation': 8,
    'Tourists': 9,
    'Government (Diplomatic)': 10,
    'Religious Figures/Institutions': 11,
    'Abortion Related': 12,
    'Journalists & Media': 13,
    'NGO': 14,
    'Telecommunication': 15,
    'Terrorists/Non-State Militia': 16,
    'Educational Institution': 17,
    'Airports & Aircraft': 18,
    'Unknown': 19,
    'Maritime': 20,
    'Food or Water Supply': 21,
    'Other': 22
}

# Map numerical values to the target type column
cleaned_terrorism_df['target_type_numeric'] = cleaned_terrorism_df['targtype1_txt'].map(target_type_values)

# Assign numerical weight to each target
target_type_weights = {
    'Business': 0.5,
    'Police': 0.7,
    'Private Citizens & Property': 0.3,
    'Utilities': 0.8,
    'Military': 0.8,
    'Violent Political Party': 0.1,
    'Government (General)': 0.8,
    'Transportation': 0.3,
    'Tourists': 0.1,
    'Government (Diplomatic)': 0.8,
    'Religious Figures/Institutions': 0.3,
    'Abortion Related': 0.1,
    'Journalists & Media': 0.5,
    'NGO': 0.5,
    'Telecommunication': 0.7,
    'Terrorists/Non-State Militia': 0.1,
    'Educational Institution': 0.6,
    'Airports & Aircraft': 0.87,
    'Unknown': 0.1,
    'Maritime': 0.7,
    'Food or Water Supply': 0.7,
    'Other': 0.1
}

# Map weights to the target type column
cleaned_terrorism_df['target_type_weight'] = cleaned_terrorism_df['targtype1_txt'].map(target_type_weights)
def compute_attack_metrics(dataframe, parameters):
    """
    Function to compute essential metrics for each incident based on specified parameters.
    
    Parameters:
    dataframe : DataFrame
        DataFrame containing the relevant data for terrorist incidents.
    parameters : dict
        Dictionary containing parameters for computing attack metrics.
        Example parameters:
        {
            'severity_factor': 0.8,
            'impact_factor': 0.5,
            # Add more parameters as needed
        }
        
    Returns:
    DataFrame
        DataFrame with computed attack metrics added as new columns.
    """
    # Compute attack severity based on the number of fatalities (nkill) and specified severity factor
    dataframe['attack_severity'] = dataframe['nkill'] + dataframe['nwound'] / parameters['severity_factor']
    
    # Compute potential impact based on the number of fatalities (nkill) and injuries (nwound)
    # and the specified impact factor
    dataframe['potential_impact'] = dataframe['nkill'] + cleaned_terrorism_df['target_type_weight'] * parameters['impact_factor']
    
    return dataframe

# Example parameters
parameters = {
    'severity_factor': 3.0,  # Adjust this factor based on your analysis requirements
    'impact_factor': 0.5  # Adjust this factor based on your analysis requirements
}

# Call the function with the cleaned_terrorism_df DataFrame and parameters
result_df = compute_attack_metrics(cleaned_terrorism_df, parameters)

# Display the resulting DataFrame with computed attack metrics
result_df.head()

# Create a DataFrame with the specified columns
data = {
    'iyear': [2023],
    'country_txt': ['United States'],
    'region_txt': ['North America'],
    'latitude': [40.7128],
    'longitude': [-74.0060],
    'attacktype1_txt': ['Bombing/Explosion'],
    'targtype1_txt': ['Business'],
    'weaptype1_txt': ['Explosives'],
    'success': [1],
    'nkill': [10],
    'nwound': [20],
    'gname': ['Fake Group'],
    'attack_severity': [10],  # Assuming a fixed value for demonstration
    'potential_impact': [30],  # Assuming a fixed value for demonstration
    'target_type_numeric': [1],  # Assuming a fixed value for demonstration
    'target_type_weight': [0.5]  # Assuming a fixed value for demonstration
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
df
def compute_attack_metrics(dataframe, parameters):
    """
    Function to compute essential metrics for each incident based on specified parameters.
    
    Parameters:
    dataframe : DataFrame
        DataFrame containing the relevant data for terrorist incidents.
    parameters : dict
        Dictionary containing parameters for computing attack metrics.
        Example parameters:
        {
            'severity_factor': 0.8,
            'impact_factor': 0.5,
            # Add more parameters as needed
        }
        
    Returns:
    DataFrame
        DataFrame with computed attack metrics added as new columns.
    """
    # Compute attack severity based on the number of fatalities (nkill) and specified severity factor
    df['attack_severity'] = df['nkill'] + df['nwound'] / parameters['severity_factor']
    
    # Compute potential impact based on the number of fatalities (nkill) and injuries (nwound)
    # and the specified impact factor
    df['potential_impact'] = df['nkill'] + df['target_type_weight'] * parameters['impact_factor']
    
    return dataframe

# Example parameters
parameters = {
    'severity_factor': 3.0,  # Adjust this factor based on your analysis requirements
    'impact_factor': 0.5  # Adjust this factor based on your analysis requirements
}

# Call the function with the cleaned_terrorism_df DataFrame and parameters
result_df1 = compute_attack_metrics(df, parameters)

# Display the resulting DataFrame with computed attack metrics
result_df1.head()
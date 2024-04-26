# Import libraies
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
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

# Load the dataset
terrorism_df = pd.read_csv(r"C:\Users\shric\Desktop\Dai\articles\CAPSTONE\effective_terrorism_tactics_exploration\data\globalterrorismdb_0522dist.csv")

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

# Annotate the peak year
plt.annotate('Peak 2014', xy=(2014, 16000), xytext=(2010, 15900), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', linewidth=3))

# Adjust plot limits to ensure the arrow is visible
plt.ylim(bottom=0)  # Set the lower limit of the y-axis to 0
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
log_model_columns = country_dummies[selected_columns]
log_model_columns
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
classification_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report)
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

# Get the top 10 countries with the highest number of fatalities and wounded
fatalities_by_country = filtered_df.groupby('country_txt')['nkill'].sum().sort_values(ascending=False).head(10)
wounded_by_country = filtered_df.groupby('country_txt')['nwound'].sum().sort_values(ascending=False).head(10)

# Create a DataFrame with the top 10 countries and their fatalities and wounded
top_countries_df = pd.DataFrame({
    'Fatalities': fatalities_by_country,
    'Wounded': wounded_by_country
})

# Plot the stacked bar chart
top_countries_df.plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
plt.title('Fatalities and Wounded by Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Number of Incidents')

# Show plot
plt.xticks(rotation=45, ha='right')
plt.legend(title='Incident Type')
plt.tight_layout()
plt.show()
# Group the data by region and calculate the total number of incidents, fatalities, and wounded for each region
grouped_by_region = filtered_df.groupby('region_txt').agg({
    'iyear': 'count',  # Total number of incidents
    'nkill': 'sum',    # Total number of fatalities
    'nwound': 'sum'    # Total number of wounded
}).reset_index()

# Rename the columns for better clarity
grouped_by_region.columns = ['Region', 'Total Incidents', 'Total Fatalities', 'Total Wounded']

# Display the grouped data
print(grouped_by_region)
grouped_by_region.describe()
grouped_by_region.head()

# Initialize LabelEncoder
label_encoder3 = LabelEncoder()

# Fit and transform the Region column
grouped_by_region['Region_Code'] = label_encoder3.fit_transform(grouped_by_region['Region'])

# Print the mapping of encoded values to Region categories
print("Encoded values for Region:")
for i, region_code in enumerate(label_encoder3.classes_):
    print(f"{region_code}: {i}")
combined_casualties = grouped_by_region['Total Fatalities'] + grouped_by_region['Total Wounded']

# Split data into features (X) and target variable (y)
X1 = grouped_by_region.drop(columns=['Region', 'Total Fatalities', 'Total Wounded'])
y1 = grouped_by_region['Total Fatalities'] + grouped_by_region['Total Wounded']

# Split data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=5)

# Fit logistic regression model
logistic_model1 = LogisticRegression(max_iter=1000)
logistic_model1.fit(X1_train, y1_train)

# Predict on the test set
y1_pred = logistic_model1.predict(X1_test)

# # Evaluate model performance
# accuracy1 = accuracy_score(y1_test, y1_pred)
# classification_rep1 = classification_report(y1_test, y1_pred)
# roc_auc1 = roc_auc_score(y1_test, y1_pred)

# print("Accuracy:", accuracy1)
# print("Classification Report:\n", classification_rep1)
# print("ROC AUC Score:", roc_auc1)
from sklearn.utils import resample

# Number of bootstrap iterations
n_iterations = 100

# Initialize lists to store evaluation metrics
accuracy_scores1 = []
classification_reports1 = []
roc_auc_scores1 = []

# Perform bootstrapping
for i in range(n_iterations):
    # Resample the data
    X_boot, y_boot = resample(X1_train, y1_train, replace=True, random_state=i)
    
    # Fit logistic regression model
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_boot, y_boot)
    
    # Predict on the test set
    y1_pred = logistic_model.predict(X1_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y1_test, y1_pred)
    # new_classification_report1 = classification_report(y1_test, y1_pred)
    # roc_auc = roc_auc_score(y1_test, y1_pred)
    
    # Store evaluation metrics
    accuracy_scores1.append(accuracy)
    # classification_reports1.append(new_classification_report1)
    # roc_auc_scores1.append(roc_auc)

# Aggregate results
mean_accuracy = np.mean(accuracy_scores1)
mean_roc_auc = np.mean(roc_auc_scores1)

# Print aggregated results
print("Mean Accuracy:", mean_accuracy)
print("Mean ROC AUC Score:", mean_roc_auc)
# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y1_test, y1_pred, color='blue')
plt.plot([min(y1_test), max(y1_test)], [min(y1_test), max(y1_test)], color='red', linestyle='--')
plt.title('Actual vs. Predicted Total Casualties')
plt.xlabel('Actual Total Casualties')
plt.ylabel('Predicted Total Casualties')
plt.grid(True)
plt.show()

# Plot residuals
residuals = y1_test - y1_pred
plt.figure(figsize=(10, 6))
plt.scatter(y1_test, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Actual Total Casualties')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Initialize LabelEncoder
label_encoder4 = LabelEncoder()

# Fit and transform the Region column
filtered_df['attack'] = label_encoder4.fit_transform(filtered_df['attacktype1_txt'])

# Print the mapping of encoded values to Region categories
print("Encoded values for Region:")
for i, attack in enumerate(label_encoder4.classes_):
    print(f"{attack}: {i}")
anova_data = filtered_df['nkill'] / filtered_df['attack']
anova_data = anova_data.dropna()
anova_data
downsampled_new_filtered_df = new_filtered_df.sample(n=len(anova_data), replace=False, random_state=42)

# Check the size of both datasets
print("Size of anova_data:", len(anova_data))
print("Size of downsampled_new_filtered_df:", len(downsampled_new_filtered_df))
# Replace inf values with a large number (e.g., 9999)
anova_data.replace(np.inf, 9999, inplace=True)

# Replace 0.000000 values with the mean of non-zero values
non_zero_mean = anova_data[anova_data != 0].mean()
anova_data.replace(0, non_zero_mean, inplace=True)
# Reset index of downsampled_new_filtered_df
downsampled_new_filtered_df.reset_index(drop=True, inplace=True)

# Reset index of anova_data and convert it to a DataFrame
anova_data = anova_data.reset_index(drop=True).to_frame(name='anova_data')

# Concatenate the anova_data DataFrame with downsampled_new_filtered_df
data_with_anova = pd.concat([downsampled_new_filtered_df, anova_data], axis=1)

# Fit ordinary least squares (OLS) model
model = ols('anova_data ~ C(region_txt)', data=data_with_anova).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Print ANOVA table
print(anova_table)
grouped_means = data_with_anova.groupby('region_txt')['anova_data'].mean()
grouped_std_errors = data_with_anova.groupby('region_txt')['anova_data'].sem()

# Print the results
print("Mean anova_data for each region:")
print(grouped_means)
print("\nStandard error of the mean for each region:")
print(grouped_std_errors)
# Mean values for each region
means = [4265.762947, 3537.855924, 3628.329621, 3248.102675, 3446.824528,
         3425.146962, 3629.673709, 3384.104235, 3419.801495, 3425.498489,
         3489.941399, 3491.350464]

# Standard errors for each region
std_errors = [815.989254, 335.044175, 268.301607, 260.392115, 61.475001,
              17.701651, 157.830276, 141.701806, 36.818876, 32.009894,
              73.468622]

# Define the x-axis labels (regions)
regions = ['Australasia & Oceania', 'Central America & Caribbean', 'Central Asia',
           'East Asia', 'Eastern Europe', 'Middle East & North Africa', 'North America',
           'South America', 'South Asia', 'Southeast Asia', 'Sub-Saharan Africa',
           'Western Europe']

## Correcting the number of standard errors to match the number of regions
std_errors.append(std_errors[-1])

# Plot the means with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(regions, means, yerr=std_errors, fmt='o', capsize=5)
plt.xlabel('Region')
plt.ylabel('Mean Fatality Rate')
plt.title('Mean Fatality Rate for Each Region with Error Bars')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()
cleaned_terrorism_df.head()
# Calculate the total number of attacks and fatalities per year
attacks_fatalities_by_year = cleaned_terrorism_df.groupby(['iyear']).agg({'success': 'count', 'nkill': 'sum', 'nwound': 'sum'}).reset_index()

# Visualize both the total number of attacks and fatalities over time
plt.figure(figsize=(12, 6))

# Plot the total number of attacks over time
plt.plot(attacks_fatalities_by_year['iyear'], attacks_fatalities_by_year['success'], color='blue', marker='o', label='Number of Attacks')

# Plot the total number of fatalities over time
plt.plot(attacks_fatalities_by_year['iyear'], attacks_fatalities_by_year['nkill'], color='red', marker='s', label='Number of Fatalities')

# Plot the total number of wounded over time
plt.plot(attacks_fatalities_by_year['iyear'], attacks_fatalities_by_year['nwound'], color='green', marker='^', label='Number of Wounded')

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Total Number of Attacks, Fatalities, and Wounded Over Time')

# Add legend
plt.legend(loc='upper left')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()
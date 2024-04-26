# Predictive Analysis of Terrorist Activities

<p align="center">
  <img src="IMG/Screenshot 2024-04-26 150044.png" width = 900 height = 60>
</p>

DAI COHORT 8 - Chris Thompson

Capstone Project: Exploring Global Terrorism Database and Making Predictions

This repository contains code for predictive analysis of terrorist activities using machine learning techniques. The analysis includes data cleaning, visualization, hypothesis testing, and building machine learning models for various classification and regression tasks related to terrorist incidents.

Dataset
The dataset used for this analysis is the Global Terrorism Database (GTD), which provides comprehensive information on terrorist incidents worldwide. The dataset includes various attributes such as the location, date, type of attack, target type, weapons used, and casualty counts.

Data
The analysis is performed on the Global Terrorism Database (GTDB) dataset.
The dataset is loaded from globalterrorismdb_0522dist.csv.
Relevant columns for analysis include:
 - iyear: Year of the terrorist attack
 - country_txt: Country where the attack occurred
 - region_txt: Region where the attack occurred
 - latitude, longitude: Geographic coordinates of the attack
 - attacktype1_txt: Type of attack
 - targtype1_txt: Type of target
 - weaptype1_txt: Type of weapon used
 - success: Success of the attack
 - nkill: Number of fatalities
 - nwound: Number of injuries
 - gname: Perpetrator group name

Analysis Steps
1. Data Cleaning:
 - Selected relevant columns and filtered data for the specified period (2000 - 2023).
2. Data Visualization:
 - Visualized top countries by number of attacks.
 - Analyzed the number of each type of attack and visualized top countries by number of attacks and top types of attack.
 - Investigated attacks over time and visualized the trend.
 - Explored the number of fatalities by country.
3. Hypothesis Testing:
 
 - Hypothesis: Certain factors significantly influence the success of a terrorist attack.
  - Null Hypothesis (H0): The features do not affect the success of a terrorist attack.
  - Alternative Hypothesis (Ha): The features have a significant impact on the success of a terrorist attack.
 - Analysis Model: Logistic regression to identify the most influential features on attack success.
 
 - Hypothesis: Different features contribute to the classification of terrorist attack types.
  - Null Hypothesis (H0): The features are not predictive of the type of terrorist attack.
  - Alternative Hypothesis (Ha): There are significant associations between certain features and the type of terrorist attack.
 - Analysis Model: Multiclass logistic regression to identify the features most associated with each attack type.
 
 - Hypothesis: Various factors contribute to the number of fatalities or injuries in a terrorist attack.
  - Null Hypothesis (H0): The selected features do not influence the number of fatalities or injuries in a terrorist attack.
  - Alternative Hypothesis (Ha): The selected features have a significant impact on the number of fatalities or injuries in a terrorist attack.
 - Analysis Model: Random forests regression to determine the most important features affecting the severity of the attack.
4. Machine Learning Models:
 - Binary Classification of Attack Success: Predicted whether a terrorist attack is successful or not using logistic regression.
 - Multiclass Classification of Attack Type: Predicted the type of attack based on various features using logistic regression.
 - Regression on Fatalities or Injuries: Predicted the number of fatalities or injuries caused by a terrorist attack using a random forest regressor.

Key Findings
The chi-square test results suggest a significant relationship between the type of attack and the region, indicating regional variations in terrorist tactics.
ANOVA results reveal differences in the severity of attacks across regions, with certain regions experiencing higher fatality rates.
The logistic regression model achieved an accuracy of approximately 89.78% in predicting the success of terrorist attacks, with better performance in predicting positive outcomes.
Conclusion
The analysis provides insights into the distribution, severity, and predictive factors of global terrorism tactics. Regional factors play a significant role in shaping the nature and prevalence of terrorist incidents, highlighting the importance of understanding localized trends for effective counterterrorism strategies.


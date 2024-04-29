# Predictive Analysis of Terrorist Activities

<p align="center">
  <img src="IMG/Screenshot 2024-04-26 150044.png" width = 900 height = 60>
</p>

DAI COHORT 8 - Chris Thompson

Capstone Project: Exploring Global Terrorism Database and Making Predictions

This repository contains code for predictive analysis of terrorist activities using machine learning techniques. The analysis includes data cleaning, visualization, hypothesis testing, and building machine learning models for various classification and regression tasks related to terrorist incidents.

This code performs an extensive analysis of global terrorism data from 2000 to 2023. It involves data cleaning, visualization, statistical analysis, machine learning modeling, and geospatial mapping. The dataset includes information such as the year of the incident, country, region, attack type, target type, weapon type, number of fatalities, and number of injuries.

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


The visualizations provide insights into various aspects of terrorism, including the top countries and regions affected, the distribution of attack types and target types, trends over time, and fatalities by country. Statistical tests like chi-square and ANOVA are used to analyze relationships between categorical variables. Machine learning models, such as logistic regression and random forests, are employed to predict the success of terrorist attacks based on various factors.

The graphs are important as they provide a visual representation of the data, making it easier to identify patterns, trends, and relationships. For example, the bar charts show the distribution of attacks across countries and the prevalence of different attack types, while the heatmap provides a spatial view of attack locations over time.

Analysis Steps
1. Data Cleaning:
 - Selected relevant columns and filtered data for the specified period (2000 - 2023).

2. Data Visualization:
 - Visualized top countries by number of attacks.

 <p align="center">
  <img src="IMG/Top 10 Countries by Attack.png" width = 500 height = 600>
</p>

 - Analyzed the number of each type of attack and visualized top countries by number of attacks and top types of attack.

<p align="center">
  <img src="IMG/Top 10 Countries by Attack Type.png" width = 500 height = 600>
</p>

 - Investigated attacks over time and visualized the trend.

<p align="center">
  <img src="IMG/Number of Attacks Over Time.png" width = 500 height = 600>
</p>

 - ROC Curve for predictions

 <p align="center">
  <img src="IMG/ROC Curve.png" width = 500 height = 600>
</p>

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

Stats and probability are used to analyze relationships between variables, assess model performance, and make predictions. For instance, chi-square tests are used to assess the independence of attack types and regions, while logistic regression models predict the success of attacks based on various features.

The compute_attack_metrics function calculates essential metrics for each incident based on specified parameters. It computes attack severity and potential impact, where severity is determined by the number of fatalities and injuries relative to a severity factor, and potential impact incorporates a weighted factor based on the target type of the incident. This weighting allows for a more nuanced assessment of the potential impact of different types of attacks.

Key Findings
Based on the provided code and analysis, here are some key findings:

Top Countries by Number of Attacks:
 - Iraq, Pakistan, and Afghanistan are the top three countries with the highest number of terrorist attacks.

Top Types of Attacks:
 - Bombing/Explosion and Armed Assault are the most prevalent types of terrorist attacks globally.

Target Types and Regions:
 - Different regions have varying patterns of target types in terrorist attacks.
   For example, in the Middle East & North Africa region, Private Citizens & Property are the most common targets.

Number of Attacks Over Time:
 - There was a peak in the number of terrorist attacks around 2014, with a gradual decrease afterward.

Number of Fatalities by Country:
 - Iraq has experienced the highest number of fatalities due to terrorist attacks.

Geospatial Analysis:
 - The distribution of terrorist attacks varies across different regions and countries.
   Certain areas have a higher concentration of attacks, indicating potential hotspots.

Binary Classification of Attack Success:
 -The logistic regression model achieved a certain level of accuracy in predicting the success of terrorist attacks based on various features.

Multiclass Classification of Attack Type:
 - The multiclass logistic regression model was able to classify terrorist attack types to some extent based on the provided features.

Regression on Fatalities or Injuries:
 - The random forests regression model was moderately successful in predicting the number of fatalities or injuries in terrorist attacks.

These findings provide insights into the patterns and trends of terrorist activities, the factors influencing attack success, the distribution of attack types, and the severity of attacks in terms of fatalities and injuries. Further analysis and refinement of machine learning models could enhance the understanding of these phenomena and improve predictive accuracy.

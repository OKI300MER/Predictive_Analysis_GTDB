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
   
	iyear	latitude	longitude	success	nkill	nwound	target_type_numeric	target_type_weight	attack_severity	potential_impact
count	144832	143825	143825	144832	138025	132456	144832	144832	131981	138025
mean	2013.82	25.58	50.39	0.867	2.54	3.46	5.67	0.53	3.32	2.81
std	4.69	13.81	37.02	0.339	11.75	44.41	4.84	0.25	22.13	11.75
min	2000	-84.67	-158.08	0	0	0	1	0.1	0	0.05
25%	2012	13.87	37.60	1	0	0	3	0.3	0	0.3
50%	2014	32.07	44.77	1	1	0	3	0.5	1	1.15
75%	2017	34.28	70.27	1	2	3	7	0.8	3	2.35
max	2021	69.98	179.37	1	1700	10878	22	0.87	5011	1700.4


The visualizations provide insights into various aspects of terrorism, including the top countries and regions affected, the distribution of attack types and target types, trends over time, and fatalities by country. Statistical tests like chi-square and ANOVA are used to analyze relationships between categorical variables. Machine learning models, such as logistic regression and random forests, are employed to predict the success of terrorist attacks based on various factors.

The graphs are important as they provide a visual representation of the data, making it easier to identify patterns, trends, and relationships. For example, the bar charts show the distribution of attacks across countries and the prevalence of different attack types, while the heatmap provides a spatial view of attack locations over time.

Analysis Steps
1. Data Cleaning:
 - Selected relevant columns and filtered data for the specified period (2000 - 2023).

2. Data Visualization:
 - Visualized top countries by number of attacks.

 <p align="center">
  <img src="IMG/Top 10 Countries by Attack.png" width = 800 height = 500>
</p>

 - Analyzed the number of each type of attack and visualized top countries by number of attacks and top types of attack.

<p align="center">
  <img src="IMG/Top 10 Countries by Attack Type.png" width = 900 height = 500>
</p>

 - Investigated attacks over time and visualized the trend.

<p align="center">
  <img src="IMG/Total Number of Attacks vs Deaths.png" width = 800 height = 500>
</p>

 - ROC Curve for predictions

 <p align="center">
  <img src="IMG/ROC Curve.png" width = 800 height = 500>
</p>

3. Hypothesis Testing:
 
 - Hypothesis: Certain factors significantly influence the success of a terrorist attack.
  - Null Hypothesis (H0): The features do not affect the success of a terrorist attack.
  - Alternative Hypothesis (Ha): The features have a significant impact on the success of a terrorist attack.
 - Analysis Model: The logistic regression analysis demonstrates how various factors like region, type of attack, and success significantly influence the success of a terrorist attack, as indicated by the model's accuracy in predicting attack success.
 - Hypothesis: Different features contribute to the classification of terrorist attack types.
  - Null Hypothesis (H0): The features are not predictive of the type of terrorist attack.
  - Alternative Hypothesis (Ha): There are significant associations between certain features and the type of terrorist attack.
 - Analysis Model: The multiclass logistic regression analysis reveals how different features such as region, target type, and weapon type contribute to the classification of terrorist attack types, providing insights into the varied nature of terrorist activities.
 
 - Hypothesis: Various factors contribute to the number of fatalities or injuries in a terrorist attack.
  - Null Hypothesis (H0): The selected features do not influence the number of fatalities or injuries in a terrorist attack.
  - Alternative Hypothesis (Ha): The selected features have a significant impact on the number of fatalities or injuries in a terrorist attack.
 - Analysis Model: The random forest regression analysis illustrates how various factors such as region, success, and type of attack contribute to the number of fatalities or injuries in a terrorist attack, highlighting the complexity of predicting such outcomes.

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

Accuracy: 0.8979591836734694
Classification Report:               

              precision    recall  f1-score   support
         0.0       0.73      0.38      0.50      3790
         1.0       0.91      0.98      0.94     24434

    accuracy                           0.90     28224
   macro avg       0.82      0.68      0.72     28224
weighted avg       0.89      0.90      0.88     28224

ROC AUC Score: 0.6805770885026985

Mean Absolute Error: 3.2087698809967864

Mean Squared Error: 1596.1432421023383

A positive correlation between attack severity (measured by fatalities) and the weight of target types, implying that attacks targeting specific entities tend to result in more fatalities. Additionally, the chi-square test results hinted at regional disparities in the severity of terrorist attacks.

The logistic regression model unveiled a relationship between target type, attack severity, and the success of terrorist attacks.

The correlation matrix shed light on the intricate relationship between geographical coordinates and various attributes of terrorist attacks. Notably, latitude exhibited a slight positive correlation with fatalities, suggesting that attacks occurring at specific latitudes might entail more severe outcomes.

These findings provide insights into the patterns and trends of terrorist activities, the factors influencing attack success, the distribution of attack types, and the severity of attacks in terms of fatalities and injuries. Further analysis and refinement of machine learning models could enhance the understanding of these phenomena and improve predictive accuracy.

Executable Severity and Impact Function
- Definition: Severity measures the extent of harm caused by a terrorist attack, primarily focusing on the number of fatalities and injuries.

- Formula: The severity of an attack is calculated using the following formula:
  - Severity=min⁡((Number of Fatalities+Number of Injuries)×Severity Factor,Max Severity)
  - Severity=min((Number of Fatalities+Number of Injuries)×Severity Factor,Max Severity)
  - Here, the severity factor is a parameter that adjusts the weight of fatalities and injuries in determining severity. The max severity is the upper limit of severity, usually set to 100%.

- Classification: Severity is classified into four categories based on the calculated severity value:
  - Low: Severity values ranging from 0 to 5.
  - Medium: Severity values ranging from 6 to 25.
  - High: Severity values ranging from 26 to 75.
  - Extremely High: Severity values ranging from 76 to 100.


- Definition: Impact evaluates the broader consequences of a terrorist attack beyond casualties, considering factors like damage to infrastructure or targeting of key entities.

- Formula: The impact of an attack is calculated using the following formula:
  - Potential Impact=min⁡((Number of Fatalities+Number of Injuries+Target Type Weight×Impact Factor),Max Impact)
  - Potential Impact=min((Number of Fatalities+Number of Injuries+Target Type Weight×Impact Factor),Max Impact)


- Here, the impact factor is a parameter that adjusts the weight of target type in determining impact. The max impact is the upper limit of impact, usually set to 100%.

- Classification: Impact is classified into three categories based on the calculated impact value:
  - Low: Impact values ranging from 0 to 10.
  - Medium: Impact values ranging from 11 to 50.
  - High: Impact values ranging from 51 to 100.


These weights will change by location as some countries may consider religious leader more important that political or business over military

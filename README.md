# Credit_Risk_Analysis
An analysis using Machine Learning algorithms to identify credit card risk using a dataset from LendingClub.

# Overview

The purpose of this analysis is to understand how to utilize `Machine Learning` statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on **Supervised Learning**. This reason why this is called **"Supervised Learning"** is because the data includes a labeled outcome. 

To complete this analysis, we use different `Machine Learning` techniques to train and evaluate the data with unbalanced classes. The dataset has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order balance out the classifications to allow for more meaningful predictions and improve the accuracy score, we needed to employ various `Machine Learning` algorithms to resample the data. These algorithms include `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, `SMOTEENN`, `BalancedRandomForestClassifier`, and `EasyEnsembleClassifier`.

# Results

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".

![Loans](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Loans%20Statuts.png)

After loading, spliting and training the dataset, we will use severals Machine Learning model to predict high and low risk labels.

## Delivrable 1

### Oversampling
In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm.

#### Naive Random Oversampling Algorithm
- The Balance accuracy
![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Naive%20Accuracy.png)

The balanced accuracy score is 0.655, this score is higher than 0.5 but still relatively low, suggesting that the model's predictions are somewhat accurate but not highly reliable.

- The Classification report
![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Naive%20Classification.png)




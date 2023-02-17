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

- **The Balance accuracy**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Naive%20Accuracy.png)

The balanced accuracy score is 0.655, this score is higher than 0.5 but still relatively low, suggesting that the model's predictions are somewhat accurate but not highly reliable.

- **The Classification report**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Naive%20Classification.png)

- Precision: the precision for the "high_risk" class is very low (0.01), indicating that the model is not very good at identifying true positives for this class.

- Recall: The recall for the "high_risk" class is higher (0.72), indicating that the model is better at finding the true positives for this class.

### Delivrable 2

#### SMOTE Oversampling

- **The Balance accuracy**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/SMOTE%20Accuracy.png)

The balanced_accuracy_score for this prediction is 0.662, which indicates a moderate level of accuracy in correctly classifying both the high-risk and low-risk classes. It is slightly higher than the previous result of 0.655, which is a positive indication that the model is improving in its performance.

- **The Classification report**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/SMOTE%20Classification.png)

- Precision: In this case, the precision for the "high_risk" class has improved slightly to 0.01, indicating that the model is still not very good at identifying true positives for this class.

- Recall: The recall for the "high_risk" class has improved to 0.63, which is a significant improvement over the previous set of metrics.

- F1 score: The F1 score for the "high_risk" class has improved to 0.02, which is consistent with the previous set of metrics.

#### Underampling

In this section, you will test an undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. You will undersample the data using the Cluster Centroids algorithm and complete the folliowing steps:

- **The Balance accuracy**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Undersampling%20Accuracy.png)

A balanced accuracy score of 0.54 indicates that the model is only slightly better than random chance at predicting the target variable. It suggests that the model may not be well-suited to the task, and further evaluation may be needed to identify the reasons for the poor performance.

- **The Classification report**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Undersampling%20Classification.png)

- Precision:  the precision for the "high_risk" class is very low at 0.01, indicating that the model is not very good at identifying true positives for this class.

Recall: The recall for the "high_risk" class is 0.69, which is a significant improvement over the previous set of metrics.

- F1 score: The F1 score for the "high_risk" class has decreased to 0.01, indicating poor performance on this class.

#### Combination (Over and Under) Sampling

In this section, you will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms.

- **The Balance accuracy**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Combination%20Accuracy.png)

The balanced accuracy score for this result is 0.5442, which indicates a moderate level of performance. It suggests that the model is able to classify both the positive and negative classes with similar accuracy.

- **The Classification report**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Combination%20Classification.png)

The report shows that the model is still not performing well on the "high_risk" class, with very low precision and F1-score, indicating that the model is incorrectly labeling many high-risk samples as low-risk. The recall score for this class is higher, indicating that the model is correctly identifying a larger fraction of the high-risk samples, but there is still room for improvement. The specificity score for the "high_risk" class is also low, indicating that the model is incorrectly labeling many low-risk samples as high-risk, which is also contributing to the poor precision and F1-score for this class.

### Delivrable 3

#### Ensemble Learners

In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier.

##### Balanced Random Forest Classifier

**The Balance accuracy**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Balanced%20Random%20Accuracy.png)

The resulting score is 0.78, which means that the random forest classifier achieved a balanced performance on the test set.

**The Classification report**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Balanced%20Classification.png)

The model's recall for the minority class (high risk) has improved to 70%, indicating that the model can better detect high-risk loans. Additionally, the precision and F1 score for the minority class have improved to 3% and 6%, respectively, which is a significant improvement over the original model's scores of 0.01% and 0.01%. Finally, the geometric mean has also improved, indicating that the model has improved in terms of overall performance. Overall, the updated classification report suggests that the new model has made significant improvements in detecting high-risk loans

##### Easy Ensemble AdaBoost Classifier

**The Balance accuracy**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Easy%20Accuracy.png)

The resulting score is 0.93, which means that the EE_model classifier achieved a high level of balanced performance on the test set. 

**The Classification report**

![](https://github.com/Hanzian/Credit_Risk_Analysis/blob/main/Images/Easy%20Classification.png)

The recall for the high-risk class has improved to 92%, which means that the model is able to detect almost all high-risk loans. Additionally, the precision for the high-risk class has improved to 9%, and the F1 score has improved to 16%, which are much better than the previous model's scores. The geometric mean has also improved, indicating that the model is performing well overall. The updated report suggests that the new model has made significant improvements in detecting high-risk loans and has very high accuracy in predicting the low-risk class.



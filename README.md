## Introduction

This project delves into the customer segmentation techniques, specifically focusing on **K-means clustering** for unsupervised customer grouping and **Random Forest**, and **Support Vector Machines (SVM)** for supervised customer classification. While methods like **RFM (Recency, Frequency, Monetary Value)** provide valuable customer insights through purchase history analysis, they often fall short in capturing the nuances of modern customer behaviour, particularly within the context of big data characterized by its "4Vs" – Volume, Variety, Velocity, and Veracity (Agrawal, Kaur, and Singh, 2023). This project explores how K-means clustering and machine learning algorithms can address these limitations, enabling businesses to gain a more comprehensive understanding of their customer base and develop targeted strategies for sustainable growth.

## **Dataset Description**

This dataset (URL: https://www.kaggle.com/datasets/carrie1/ecommerce-data ) from the UCI Machine Learning Repository offers a valuable window into customer behaviour within an online retail setting. It comprises real transaction data from a UK-based retailer specializing in unique gifts, spanning a period from December 1st, 2010, to September 12th, 2011. The dataset presents an opportunity to explore customer segmentation, analyse customer lifetime value, predict customer churn, and delve into product popularity trends. While it represents a single retailer, the insights gained can inform effective marketing strategies across the eCommerce industry.

## **EDA**

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/1c50319c-1eb0-44bc-a6e8-7f5e03b67cdd)

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/cd7e9d14-7b54-4cad-8f03-e5f6b5bcaea0)

The dataset contains **541,909 transactions(rows)** with **8 features(columns)** including product details (code, description), quantity, price, customer ID (with missing values), transaction date, and customer country. Description and customer ID have missing values that need to be addressed. The date format allows for time series analysis, and customers can have multiple transactions. Further cleaning and feature creation are needed before proceeding.

## Data Preprocessing

1.  Handle missing values

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/8d8aba32-5637-4fb8-ac91-52e8ac7b03e1)

There are 135,080 missing values (24.93%) exist in CustomerID, causing issues for clustering and recommendations. Removing rows with missing IDs is preferred to maintain data integrity. And there are few 1,454 missing values (0.27%) present in Description. However, inconsistencies exist where the same product code has different descriptions. Removing rows with missing descriptions avoids propagating errors and inconsistencies.

2.  Handle duplicated values

The dataset contains duplicate rows (5,225 rows) with identical transaction times. These likely represent recording errors, not real repurchases. Keeping them can skew recommendations and clustering.  Removing duplicates will create a cleaner dataset for building accurate customer profiles and product recommendations.

3.  Handle cancelled transactions.

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/fbf67a33-3d64-4eb3-800d-c2f0fe02597f)

Cancelled transactions (invoice numbers starting with "C" and negative quantities) involve a variety of products. To understand customer cancellation behaviour and improve recommendations, these transactions will be kept. This can help identify cancellation patterns as buying habits and potentially avoid recommending frequently cancelled products.

4.  Handle Stock code and  UnitPrice anomalies.

Out of the 3,684 unique stock codes in the dataset, there are a total of 8 anomalies where the character count is either 0 or 1. 
After reviewing the descriptions of the respective codes, it appeared that they did not resemble product codes, and thus were removed. The dataset exhibits anomalies in stock codes, comprising approximately 0.48% of the total records. Following the elimination of these anomalies, the dataset now consists of 399,689 rows.
There are rows with a unit price of 0. Thirty-three such rows were identified and given their potential to introduce noise in machine learning algorithms, they were removed from the dataset, accounting for 33 transactions.

1. Data preprocessing Results
**Total 339,656 rows.**

## **Feature Engineering**

### **RFM**

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/84f9ec4e-ee7c-49d1-b760-85c737eb0ca8)

### **Handle Outlier/ Anomaly Detection**

After employing the IsolationForest model for outlier detection, outliers were removed from the dataset. Initially, setting the contamination parameter to 'auto' led to significant data loss. Thus, the parameter was manually adjusted to 0.05 to address this issue.

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/b8f87bc3-5a33-4323-be75-f5d02e6f10ab)f5caea5ee4e/Untitled.png)

### **Correlation analysis**

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/2d6ddf94-1e81-4556-ad5b-31cfedd04a31)

Examining the heatmap, it's evident that certain pairs of variables exhibit strong correlations. Notably:

- Totals spend and total transactions.
- Cancellation frequency and cancellation rate

These correlations are significant.

### PCA
PCA offers numerous benefits: mitigating multicollinearity by removing redundant features, enhancing K-means clustering accuracy by simplifying data, filtering out noise for more stable clusters, improving visualization of customer segments, and speeding up computational processes for efficiency. First, all features were normalized to have a mean of 0 and a standard deviation of 1, except for Customer ID.
![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/c1399769-4608-4d79-960d-8976bca65224)

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/31a8d509-5011-41b9-bce4-fc146399fc9d)
The plot and cumulative explained variance values illustrate how each principal component contributes to capturing the dataset's total variance. Notably, the first component explains around 43% of the variance, while the first two components together explain about 43%. As we add components, the cumulative variance increases, with a notable slowdown after the 4th component, capturing approximately 81% of the total variance. Retaining these first 4 components strikes a balance, effectively reducing dimensionality while retaining sufficient information for customer segmentation.

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/df37fed7-3d94-4297-8a48-c7c1ff2b13c6)

## **K-means Clustering**
K-Means stands as an unsupervised machine learning technique, dividing data into a predefined number of clusters (K) through the minimization of within-cluster sum-of-squares (WCSS), also termed inertia. This process entails iteratively assigning data points to their closest centroid and adjusting centroids based on the mean of assigned points until convergence or the fulfilment of a specified stopping criterion. To pinpoint the ideal number of clusters (k) for customer segmentation, I will apply two widely recognized approaches: the Elbow Method and the Silhouette Method. Employing both methodologies is a standard procedure for validating outcomes.
### Elbow Method

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/423aae64-5a6b-4d25-ad8b-d91f7697311a)
The Elbow Method suggests an optimal k value of 4 for the K-means clustering algorithm, although the elbow point is not very distinct, which is common in real-world data. Inertia continues to decrease notably up to k=5, indicating the optimal k could be between 3 and 7.

### Silhouette Method
The Silhouette Method evaluates the quality of clusters by gauging the adequacy of data points within their respective clusters and the differentiation between clusters. It computes silhouette coefficients for individual points, spanning from -1 to 1, where higher values denote enhanced clustering. The mean silhouette score offers a comprehensive assessment, with elevated scores indicating superior clustering performance.

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/1b75fc72-75a4-43ca-b35f-eba0fed19cb1)
To interpret silhouette plots and determine the optimal number of clusters (k), consider the following guidelines: analyse silhouette score widths (wide indicates well-separated clusters, narrow indicates poorly defined clusters), assess uniformity in cluster size, look for clear peaks in average silhouette scores, minimize fluctuations in silhouette plot widths, maximize the overall average silhouette score, and visually inspect silhouette plots for consistent cluster formation and compactness. Based on these guidelines, selecting k=4 is preferable, as it yields more balanced and well-defined clusters, strengthening the clustering solution.

### Clustering Evaluation

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/165fcdc0-1082-4b6c-9ae3-16975398a4f4)
The distribution of customers among the clusters, illustrated in the bar plot, indicates that clusters 0 and 1 contain approximately 52.9% and 31.43% of customers, respectively, while cluster 2 accommodates around 10.47% of customers. Cluster 3 represents 5.21% of customers.

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/9a3c998d-ccb3-4c74-900c-0f09ffc8f862)
The Silhouette Score, approximately 0.319, though not approaching 1, indicates a moderate level of separation between clusters, suggesting some distinctiveness with possible minor overlaps. Ideally, a score closer to 1 signifies more distinct and well-separated clusters.
The Calinski-Harabasz Score stands at approximately 1649.143, considerably high, indicating well-defined clusters. Higher scores in this metric generally denote better cluster definitions, suggesting significant structure in the data.
With a Davies-Bouldin Score of 1.12, we observe a reasonable level of similarity between each cluster and its most similar one. Lower scores are preferable, indicating less similarity between clusters. Our score suggests a moderate separation between clusters.
Overall, these metrics indicate good quality clustering, with well-defined and moderately separated clusters. However, there's potential for further optimization, possibly through exploring alternative clustering and dimensionality reduction algorithms.

### SVM Classifier

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/e2ab0297-59c8-46f2-bc17-526373376192)
The non-normalized confusion matrix displays the raw counts of predictions. For example, it shows that for class 0, there were 868 true positive predictions, 3 false positive predictions, and no false negatives or true negatives. 
![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/01f2cd2f-b71b-4101-a186-45f99c96cffa)
The normalized confusion matrix expresses these counts as proportions relative to the true class sizes. For instance, in the normalized matrix, a value of 0.996 in row 1, column 1 indicates that 99.6% of actual class 0 instances were correctly predicted as class 0.
both matrices highlight the classifier's strong performance across all classes, with high numbers of true positives and minimal false positives and false negatives.

a.	Precision: The precision measures the accuracy of positive predictions. For all classes, the precision is very high, ranging from 0.99 to 1.00. This indicates that when the classifier predicts a certain class, it is highly likely to be correct.
b.	Recall: Recall, also known as sensitivity, measures the ability of the classifier to correctly identify all relevant instances. Like precision, recall is very high for all classes, ranging from 0.98 to 1.00. This indicates that the classifier effectively captures most instances of each class.
c.	F1-score: The F1-score is the harmonic mean of precision and recall and provides a balance between the two metrics. Like precision and recall, the F1-score is also very high for all classes, ranging from 0.98 to 1.00. This indicates robust performance across all classes.
d.	Accuracy: The overall accuracy of the classifier is 0.99, indicating that it correctly predicts the class for 99% of the instances in the dataset.
e.	Cross-validation


### Random Forest

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/4ab8fd1b-c2b6-454c-83d3-c83195f5eff8)
The non-normalized confusion matrix displays the raw counts of predictions. For example, it shows that for class 0, there were 854 true positive predictions, 16 false positive predictions, and 1 false negative prediction. 

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/a38d1652-7257-45bb-b9ef-8d13ad1b70e2)
The normalized confusion matrix expresses these counts as proportions relative to the true class sizes. For instance, in the normalized matrix, a value of 0.980 in row 1, column 1 indicates that 98.0% of actual class 0 instances were correctly predicted as class 0.

a.	Precision: For class 0, the precision is 0.98, indicating that 98% of instances predicted as class 0 are class 0. Similarly, for classes 1, 2, and 3, the precision values are 0.95, 0.95, and 0.97, respectively.
b.	Recall: Recall, or sensitivity, represents the proportion of actual instances that were correctly predicted by the model. For class 0, the recall is 0.98, meaning that 98% of actual class 0 instances were correctly predicted. The recall values for classes 1, 2, and 3 are 0.96, 0.97, and 0.88, respectively.
c.	F1-score: The F1-score is the harmonic mean of precision and recall, providing a balance between them. The F1-scores for classes 0, 1, 2, and 3 are 0.98, 0.95, 0.96, and 0.92, respectively.
d.	Accuracy: The overall accuracy of the model is 0.97, indicating that it correctly predicts the class for 97% of instances in the dataset.
e.	Cross-validation


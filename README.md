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

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/c1399769-4608-4d79-960d-8976bca65224)

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/31a8d509-5011-41b9-bce4-fc146399fc9d)

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/df37fed7-3d94-4297-8a48-c7c1ff2b13c6)

## **K-means Clustering**

### Elbow Method

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/423aae64-5a6b-4d25-ad8b-d91f7697311a)

### Silhouette Method

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/1b75fc72-75a4-43ca-b35f-eba0fed19cb1)

### •  Clustering Evaluation

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/165fcdc0-1082-4b6c-9ae3-16975398a4f4)

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/9a3c998d-ccb3-4c74-900c-0f09ffc8f862)

### SVM Classifier

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/e2ab0297-59c8-46f2-bc17-526373376192)
![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/01f2cd2f-b71b-4101-a186-45f99c96cffa)

### Random Forest

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/4ab8fd1b-c2b6-454c-83d3-c83195f5eff8)

![image](https://github.com/Youyeon-Kim/customer_segmentation_ecommerce/assets/60176735/a38d1652-7257-45bb-b9ef-8d13ad1b70e2)

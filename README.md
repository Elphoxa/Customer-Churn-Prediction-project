# **Project README**
# **Customer-Churn-Prediction-project**
Developing a machine learning model that predict whether a customer is likely to churn or not

## **Project Overview**
This project aims to assist a telecommunication company, specifically Vodafone Corporation, in addressing the critical challenge of customer churn. Customer retention is essential for sustaining business growth, as acquiring new customers is often more costly than retaining existing ones. By leveraging machine learning techniques, the project seeks to analyze historical customer data, identify key indicators of churn, and develop effective retention strategies.

## **Table of Contents**
* Business Understanding
* Data Understanding
* Data Preparation
* Model Development
* Retention Strategies
* Communication of Results

## **Business Understanding**
As a leading telecommunication corporation, Vodafone Corporation recognizes the criticality of customer retention in sustaining business growth. The escalating issue of customer churn can have a detrimental impact on revenue and profitability. The primary objective is to develop robust machine-learning models to predict customer churn accurately and formulate targeted retention strategies to achieve higher profitability.

## **Data Understanding**
Target Variable: Churn
Binary variable (Yes/No) indicating whether a customer has churned or not.

### **Features:**
#### **Demographic Information**
* Gender
* SeniorCitizen
* Partner
* Dependents

#### **Service-related Information**
* Phone Service
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies

#### **Contract and Billing Information**
* Contract
* PaperlessBilling
* Payment Method

#### **Usage and Financial Information**
* Tenure
* MonthlyCharges
* TotalCharges

### **Data Types:**
* Categorical variables (e.g., Gender, Partner) may require encoding.
* Numerical variables (e.g., Tenure, MonthlyCharges) may require scaling.

### **Data Exploration:**
* Explore distribution of churn.
* Visualize relationships between features and churn.
* Analyze summary statistics and potential outliers.

### **Data Preparation**
* Handling Missing Values
* Data Encoding
* Feature Scaling
* Exploratory Data Analysis (EDA)

### Model Development
    Train-Test Split
    Model Selection (e.g., Logistic Regression, Decision Trees, Random Forest)
    Model Training
    Model Evaluation (Accuracy, Precision, Recall, F1 Score)

### Key Features Influencing Churn
* Tenure
* Contract
* MonthlyCharges
* TotalCharges
* InternetService
* OnlineSecurity
* TechSupport
* PaperlessBilling
* PaymentMethod
* Additional Services (OnlineBackup, DeviceProtection, StreamingTV, StreamingMovies)
* Demographic Information (Gender, SeniorCitizen, Partner, Dependents)

## **Hypotheses**
* **Null Hypothesis (H0):** There is no significant difference in churn rates between customers with longer contract terms and those using the payment method (Automatic).

* **Alternative Hypothesis (H1):** There is a significant difference in churn rates between customers with longer contract terms and those using the payment method (Automatic).

## **Questions**
1. How do contract terms and payment methods correlate with customer churn?

2. Are there specific services that significantly impact churn rates?

3. Are there specific services that customers with longer contract terms tend to use more frequently?

4. Do customers using automatic payment methods show different churn patterns compared to other payment methods?

5. Are senior citizens more or less likely to churn compared to non-senior citizens?


# AUTHOR
Efosa Dave Omosigho
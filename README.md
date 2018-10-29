---
# Project Predictive Modeling: MVP
### Predicting future stock price returns. *(a Kaggle competition from sigma2)*

## Introduction

***Can we actually predict stock prices with Machine Learning?***

Investors make educated guesses by analyzing data. They'll read the news, study the company history, industry, trends...  There are lots of data points that go into malking a predction. 
The prevailing theory is that stock prices are totally random and unpredicatble. A blind-folded monkey throwing darts at a newspaper's financial pages could select a protfolio that could do just as well as one carefully selected by experts. 

But that raises the question, why do top firms like Morgan Stanley and citigroup hire quantitative analysts to go build predictive models. We have this idea of a trading floor filled with adrenalined-infused men, with loose ties, running around, yelling something into a phone. But these days are more likely to see rows of ML experts quietly sitting in front of computer screens in fact about 70% of all orders on Wall street are now placed by software. 

In this competition, I must predict a signed confidence value, ŷ ti∈[−1,1] , which is multiplied by the market-adjusted return of a given assetCode over a ten day window. If a stock is expected to have a large positive return *compared to the broad market* over the next ten days, a large, positive confidenceValue (near 1.0) should be assigned. If the stock is expected to have a negative return, a large, negative confidenceValue (near -1.0) must be assigned. If unsure, assign it a value near zero.

## Question: 

This kaggle competition aims to predict stock price performance by extracting features from pieces of news. 

## Datasets

This competition is only supported using the Kaggle kernel environment (i.e., we cannot use our PC notebook or other IDE environment). 
Kaggle provides 2 csv files, one with all the necessary market data and the other with the necessary news information.

#### Market

This dataset contains market data from February 2007 to December 2016

- Data Set Characteristics:
    - Number of Instances: 4072956
    - Number of Attributes: 16 types (13 numeric, 2 categorical (or text) and 1 datetime)
    - Missing Attribute Values: few values from features: returnsClosePrevMktres1, returnsOpenPrevMktres1, returnsClosePrevMktres10, returnsOpenPrevMktres10
    - Donor: Market data provided by Intrinio. 

The **target label** is returnsOpenNextMktres10. In the training set for date t, this is the return from t+1 market open to t+10 market open.

#### News

Contains news articles/alerts data from January 2007 to December 2016

- Data Set Characteristics:
    - Number of Instances: 9328750
    - Number of Attributes: 35 types (15 numeric, 11 categorical and 3 boolean)
    - Missing Attribute Values: Unknown due to memory limit
    - Donor: News data provided by Thomson Reuters. Copyright ©, Thomson Reuters, 2017. All Rights Reserved.


## Schema

- **Data Acquisition**
    - Import the module and create an environment within Kaggle's kernel
    - Get the training data into dataframes
    - Features briefing. 
- **Preprocessing**
    - Clipping target variable to be between 0 and 1
    - Normalization:
    - Trimming dataset from useless columns
    - Prep the news and market tables to be merged into one. 
- **EDA**
- **Feature Selection**
- **Modelization**
    - Dataset division in training and test. 
    - Optimization (training) approaches:
- **Prediction**
    - Predicting returnsOpenNextMktres10
    - *get_prediction_days* is a generator which loops through each day and provides all market and news observations which occurred since the last data you've received. 
- **Evaluation**
- **Results submission**
    - predictions_df: DataFrame which must have the following columns:
        - assetCode: The market asset.
        - confidenceValue: Your confidence whether the asset will increase or decrease in 10 trading days. All values must be in the range [-1.0, 1.0].
    - Store your predictions for the current prediction day with the kaggle function *predict*
    - write_submission_file

## Results 

Logit regressors are commonly used to estimate the probability that an instance belongs to a particular class. So the prediction can be made by classifying on whether a stock will rise or sink. 

## Intial research and Results
Build a 10 day window 

---
## LIMITATIONS
- Since this is a Kernels-only, time-based competition, I'm bound to use the kaggle kernel which is not very practical nor fast. I'm bound to make sure every test I make on their kernel is correctly designed (so there is no time wasting with simple errors). This is designed to simulate the volume, timeline, and the computational burden that real future data will introduce.
- The assetCode is not guaranteed to be unique over time. Here I specifically chose AAPL.O because we all know Apple hasn't changed it's ticker symbol. But that's not guaranteed so you have to be very careful. 

---
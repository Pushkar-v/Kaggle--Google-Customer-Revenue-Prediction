# Kaggle: Google Customer Revenue Prediction

[Kaggle Link]: www.kaggle.com/c/ga-customer-revenue-prediction

## Project Definition & Introduction

### Background & Context

Google runs a merchandise where they sell Google branded goods. The store is also known as GStore. The task for the Kaggle competition is to analyze customer dataset from a GStore to predict revenue per customer. This prediction can help managers to Identify potential customers and Market to them and also plan for store inventory management for improving in-store customer experience.

### Problem Statement

The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies. In the real world, this translates to a lot of potential marketing investment not yielding the inten ded return in revenue.

Given the tools and techniques in data analysis, our task is to generate actionable operational changes and provide insights that can be used to dedicate marketing budgets more effectively not just for GStore, but also for those companies that wish to use data analysis on top of GA data.

### Key Question

Can a predictive model prove useful to GStore for closely estimating revenue generated by visitors over a certain period of time? These estimations can be used for inventory management to balance demand and supply, future planning and also by the marketing team to design promotional strategies to target people at appropriate times.

### Solution Overview

The final solution is a two step approach in which we first use a classification model and a stacked regression model to identify as well as predict customer revenue.
We first predict customer’s tendency to make a purchase in the store. We first want to know what leads to a customer completing a purchase on the store before we try to estimate how much he purchases. 

For this we try to maximize the recall score of the model so that we capture all possible purchases. Next, we use this prediction as an indicator for predicting the revenue for these customers. We filter only the customers who have made a purchase from the first prediction and predict revenue for those customers only. This helps reduce the final error (rmse) and give out a better prediction.
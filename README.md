# Customer Churn Prediction using an Artificial Neural Network (ANN)

A binary classification project that predicts whether a bank customer will leave (churn) using a neural network built with TensorFlow/Keras.

## Project Overview

Customer churn means a customer stops using a company's services. Retaining an existing customer is usually cheaper than acquiring a new one, so being able to flag customers who are likely to leave is valuable for banks.

This project trains a feed-forward neural network on customer attributes (age, balance, number of products, geography, etc.) to predict the `Exited` label (1 = churned, 0 = stayed).

## Dataset

- **Name:** Credit Card Customer Churn Prediction (`Churn_Modelling.csv`)
- **Source:** [Kaggle](https://www.kaggle.com/) (search for "Credit Card Customer Churn Prediction")
- **Size:** 10,000 customers, 14 columns
- **Target:** `Exited` (1 = customer left, 0 = customer stayed)

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has a credit card (0/1) |
| IsActiveMember | Active member (0/1) |
| EstimatedSalary | Estimated salary |

The dataset is not included in this repository. Download `Churn_Modelling.csv` from Kaggle and place it in the project folder (see "How to Run").

## Workflow

1. **Load data** with pandas.
2. **Drop irrelevant columns:** `RowNumber`, `CustomerId`, `Surname`.
3. **Encode categorical features:** one-hot encoding of `Geography` and `Gender` (`drop_first=True`), giving 11 input features.
4. **Train/test split:** 80% train, 20% test (`random_state=1`).
5. **Feature scaling:** `StandardScaler`, fitted on the training set only and then applied to the test set to avoid data leakage.
6. **Build the ANN** (Keras `Sequential`):
   - Dense(11, ReLU), input dimension 11
   - Dense(11, ReLU)
   - Dense(1, Sigmoid)
   - Total trainable parameters: 276
7. **Compile:** Adam optimizer, binary cross-entropy loss, accuracy metric.
8. **Train:** 100 epochs with a 20% validation split.
9. **Predict and evaluate:** sigmoid outputs are converted to class labels with a 0.5 threshold, then evaluated with accuracy. Loss and accuracy curves are plotted.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **~85.85%** |
| Final training accuracy | ~85.8% |
| Final validation accuracy | ~84.4% |

## Key Learnings

- Neural networks need **scaled inputs**. Training on raw features (balance and salary are in the tens of thousands) caused huge losses and unstable validation accuracy.
- The same scaler must be applied to the training and test data.


## Tech Stack

Python, NumPy, pandas, scikit-learn, TensorFlow/Keras, Matplotlib


## Project Structure

```
customer-churn-prediction/
├── CustomerChurnPrediction.ipynb   # Main notebook
├── README.md
├── requirements.txt
└── .gitignore
```



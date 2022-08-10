# Modeling, based on https://www.kaggle.com/code/sushantb1649/brain-stroke-prediction-by-sushant-bisht
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/kaggle/input/full-filled-brain-stroke-dataset/full_data.csv")

# Pre-processing
df = df.sample(frac = 1).reset_index()
df.dropna(inplace = True) # Imputation

x = df.iloc[:, :-1]
y = df.iloc[:,-1]

encoder = OrdinalEncoder()
for i in x.columns:
    if x[i].dtypes == "object":
        x[i] = encoder.fit_transform(x[i])

# Split train/test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Standardize data
scaler = StandardScaler()
x_train_scaled = scale.fit_transform(x_train) # Fit only on known examples
x_test_scaled = scale.transform(x_test)

# Testing Logistic Regression...
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

print("Accuracy for Training Data : ", model.score(x_train_scaled, y_train))
print("Accuracy for Test Data : ", model.score(x_test_scaled, y_test))

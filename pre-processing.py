# Modeling, based on https://www.kaggle.com/code/sushantb1649/brain-stroke-prediction-by-sushant-bisht
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/full-filled-brain-stroke-dataset/full_data.csv")

# Pre-processing
df = df.sample(frac = 1).reset_index()
df.dropna(inplace = True) # Imputation

x = df.iloc[:, :-1]
y = df.iloc[:,-1]



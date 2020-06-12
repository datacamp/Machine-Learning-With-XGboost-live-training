import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

bookings = pd.read_csv('../data/hotel_bookings_clean.csv')

# 0 WHAT COLUMNS DO WE HAVE?
print(bookings.head())

# 0 WHAT ARE WE TRYING TO PREDICT?
bookings['is_canceled'].value_counts().plot(kind='barh')

# 0 QUICK CHECK ON CORRELATION
correlation = bookings.corr()['is_canceled'].sort_values(ascending=False)
print(correlation)

# Test/Train Split
X, y = bookings.iloc[:,:-1], bookings.iloc[:,-1]
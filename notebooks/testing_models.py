import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

bookings = pd.read_csv('../data/hotel_bookings_clean.csv')

################ 0 Getting to know our dataset

# WHAT COLUMNS DO WE HAVE?
print(bookings.head())
print(bookings.describe())

# WHAT ARE WE TRYING TO PREDICT?
bookings['is_canceled'].value_counts().plot(kind='barh')

# QUICK CHECK ON CORRELATION
correlation = bookings.corr()['is_canceled'].sort_values(ascending=False)
print(correlation)

################ 1 Feature/Label split
X, y = bookings.iloc[:,1:], bookings.iloc[:,0]

################ 2 YOUR FIRST MODEL 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.20, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier()

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# 3 USING DMATRIX
bookings_dmatrix = xgb.DMatrix(data=X,label=y)


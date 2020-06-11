
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


bookings = pd.read_csv('../data/hotel_bookings.csv')

# clarify the meal column
bookings['meal'].replace(to_replace = 'SC', value = 'No meal', inplace = True)
bookings['meal'].replace(to_replace = 'Undefined', value = 'No meal', inplace = True)

# clean up months
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",]
i = 1
for m in months:
    bookings['arrival_date_month'].replace(to_replace = m, value = i, inplace = True)
    i += 1

# drop rows with 0 guests
no_guests = list(bookings.loc[bookings['adults'] + bookings['children'] + bookings['babies'] == 0].index)
bookings.drop(bookings.index[no_guests], inplace = True)

# change this column to be binary on whether booked by company or agent
bookings['company'] = pd.notna(bookings['company'])
bookings['agent'] = pd.notna(bookings['agent'])

# fill the child column since it has some n/as
bookings['children'].fillna(0)

# rename columns for bettter interpretability
bookings = bookings.rename(columns={"adr": "avg_daily_rate", "company": "booked_by_company","agent": "booked_by_agent"})

num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month", "arrival_date_month",
    "stays_in_weekend_nights","stays_in_week_nights","adults","children", "babies","is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces", "total_of_special_requests", "avg_daily_rate"]

cat_features = ["hotel","meal","market_segment", "distribution_channel","reserved_room_type","deposit_type","customer_type","booked_by_company","booked_by_agent"]

all_features = bookings[num_features]

for c in cat_features:
    all_features = all_features.join(pd.get_dummies(bookings[c], prefix=c))

print(all_features.info())

#features = num_features + cat_features
#X = bookings.drop(["is_canceled"], axis=1)[features]
# y = bookings["is_canceled"]
 




'''
num_transformer = SimpleImputer(strategy="constant")

# Preprocessing for categorical features:
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical features:
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features), ("cat", cat_transformer, cat_features)])

X = preprocessor.fit_transform(X)
print(X)


CLEANING
see: https://www.kaggle.com/ujansen/hotel-reservation-cancellation-prediction
https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations


# Get rid of null values
replacements = {'children': 0.0, 'country': 'Unknown', 'agent': 0.0, 'company': 0.0}
bookings.fillna(replacements, inplace = True)


# clarify the meal column
bookings['meal'].replace(to_replace = 'SC', value = 'No meal', inplace = True)
bookings['meal'].replace(to_replace = 'Undefined', value = 'No meal', inplace = True)

# drop rows with 0 guests
no_guests = list(bookings.loc[bookings['adults']
                   + bookings['children']
                   + bookings['babies'] == 0].index)
bookings.drop(bookings.index[no_guests], inplace = True)


################################
#  Introducting the dataset

# WHAT COLUMNS DO WE HAVE?
print(bookings.head())

# WHAT ARE WE TRYING TO PREDICT?
# TODO: find percents of cancellations
# maybe graph:
# cancelp=df.groupby(["hotel","is_canceled"]).lead_time.count().reset_index()
# cancelp.columns=["hotel","is_canceled","count"]
# ax = sns.barplot(x="hotel", y="count", hue="is_canceled", data=cancelp)

# QUICK CHECK ON CORRELATION
correlation = bookings.corr()['is_canceled']
# print(correlation)

################################
#  A LITTLE PREP: ONE HOT ENCODING
bookings.dtypes
cat_features = list(bookings.select_dtypes(include = [object]))
print(cat_features)

label_encoder = preprocessing.OneHotEncoder() 

''' 
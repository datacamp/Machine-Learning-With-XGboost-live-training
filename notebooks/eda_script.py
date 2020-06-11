
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bookings = pd.read_csv('../data/hotel_bookings.csv')

''' 
CLEANING
see: https://www.kaggle.com/ujansen/hotel-reservation-cancellation-prediction
''' 

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

# QUICK CHECK ON CORRELATION
correlation = bookings.corr()['is_canceled']
print(correlation)

################################
#  A LITTLE PREP: ONE HOT ENCODING
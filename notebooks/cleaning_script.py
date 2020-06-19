import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

bookings = pd.read_csv('../data/hotel_bookings.csv')


# clean up months
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",]
i = 1
for m in months:
    bookings['arrival_date_month'].replace(to_replace = m, value = i, inplace = True)
    i += 1

# drop rows with 0 guests
no_guests = list(bookings.loc[bookings['adults'] + bookings['children'] + bookings['babies'] == 0].index)
bookings.drop(bookings.index[no_guests], inplace = True)

# clarify the meal column bc 'SC' and 'Undefined' mean the same thing according to notes
bookings['meal'].replace(to_replace = 'SC', value = 'No meal', inplace = True)
bookings['meal'].replace(to_replace = 'Undefined', value = 'No meal', inplace = True)

# change this column to be binary on whether booked by company or agent
bookings['company'] = pd.notna(bookings['company']).astype(int)
bookings['agent'] = pd.notna(bookings['agent']).astype(int)

# fill the child column since it has some n/as
bookings['children'].fillna(0)

# rename columns for bettter interpretability
bookings = bookings.rename(columns={"adr": "avg_daily_rate", "company": "booked_by_company","agent": "booked_by_agent"})

clean_features = ["is_canceled", "lead_time","arrival_date_week_number","arrival_date_day_of_month", "arrival_date_month",
    "stays_in_weekend_nights","stays_in_week_nights","adults","children", "babies","is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces", "total_of_special_requests", "avg_daily_rate", "booked_by_company","booked_by_agent"]

cat_features = ["hotel","meal","market_segment", "distribution_channel","reserved_room_type","deposit_type","customer_type"]

clean_df = bookings[clean_features]

for c in cat_features:
    clean_df = clean_df.join(pd.get_dummies(bookings[c], prefix=c))

clean_df = clean_df.rename(columns={"hotel_City Hotel": "hotel_City", "hotel_Resort Hotel": "hotel_Resort",
                                        "market_segment_Offline TA/TO": "market_segment_Offlin_TA_TO",
                                        "market_segment_Online TA": "market_segment_Online_TA",
                                        "distribution_channel_TA/TO": "distribution_channel_TA_TO",
                                        "deposit_type_No Deposit": "deposit_type_No_Deposit",
                                        "deposit_type_Non Refund": "deposit_type_Non_Refund"})

clean_df.to_csv("../data/hotel_bookings_clean.csv", encoding='utf-8', index=False)


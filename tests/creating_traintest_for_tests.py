# This is only for rough work purposes, not to be used for main

import os

# create train and test datasets in csv format from dataset/Hotel_Reservations.csv
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv(os.path.join("dataset", "Hotel_Reservations.csv"))
# Drop the 'Booking_ID' column
# data.drop(columns=["Booking_ID"], inplace=True)

# just split the dataset into training and testing sets, no need to separate the target variable
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# save the train and test datasets to csv files
train_data.to_csv(os.path.join(r"tests\catalog", "train.csv"), index=False)
test_data.to_csv(os.path.join(r"tests\catalog", "test.csv"), index=False)

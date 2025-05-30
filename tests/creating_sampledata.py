# This is only to create the sample data for testing purposes, not to be used for main


import os

import pandas as pd

# create a sample of the dataset\Hotel_Reservations.csv for testing
# Load the dataset
data = pd.read_csv(os.path.join("dataset", "Hotel_Reservations.csv"))
# Drop the 'Booking_ID' column
# data.drop(columns=["Booking_ID"], inplace=True)

# create a sample of the dataset
sample_data = data.sample(frac=0.05, random_state=42)  # 5% sample

# save the sample dataset to csv file in the location test\testing_sample
sample_data.to_csv(os.path.join("tests\\testing_sample", "sample_data.csv"), index=False)

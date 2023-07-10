# Prepare a dataset of customer having the features date, price, product_id, quanti
import pandas as pd
import random
import datetime
# Define the number of records in the dataset
num_records = 30
# Generate random data for each feature
dates = pd.date_range(start='2022-01-01', end='2022-12-31', periods=num_records)
prices = [round(random.uniform(10, 100), 2) for _ in range(num_records)]
product_ids = [random.randint(1, 100) for _ in range(num_records)]
quantities = [random.randint(1, 10) for _ in range(num_records)]
serial_nos = [f'SN-{random.randint(1000, 9999)}' for _ in range(num_records)]
user_ids = ['U' + str(random.randint(10, 20)) for _ in range(num_records)]
user_types = ['Retail', 'Wholesale']
user_classes = ['Class A', 'Class B', 'Class C']
purchase_weeks = [date.isocalendar()[1] for date in dates]
# Create a dictionary with the data
data = {
'Date': dates,
'Price': prices,
'Product_ID': product_ids,
'Quantity_Purchased': quantities,
'Serial_No': serial_nos,
'User_ID': user_ids,
'User_Type': random.choices(user_types, k=num_records),
'User_Class': random.choices(user_classes, k=num_records),
'Purchase_Week': purchase_weeks
}
# Create a DataFrame from the dictionary
df = pd.DataFrame(data)
# Print the first few rows of the dataset
print(df.head())
# Save the dataset to a CSV file
df.to_csv('customer_dataset.csv', index=False)

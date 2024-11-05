import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Input file names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# Reuse this to load the train.csv and test.csv files
def dataFromCSV(fname):
    # LOAD DATASET INTO DATAFRAMES
    df = pd.read_csv(fname)
    x_data = df.copy().drop(["ID", "Price"], axis=1)
    y_data = df["Price"]
    ids = df["ID"]
    return ids, x_data, y_data


# Read the training data from the train.csv file
ids, x_data, y_data = dataFromCSV(TRAIN_FILE)

# Drop the categorical values from the dataset, need to encode these as dummy variables to use 
# in the model
x_data = x_data.drop(["Cut", "Color", "Clarity", "Symmetry", "Polish", "Report"], axis=1)

# Split the training data into two sets, training and validation, to get an idea
# of what the error rate (MAPE) and R^2 cvalue will be
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Define the regression model
reg = LinearRegression()

# Fit the regression model using the training split
reg.fit(x_train, y_train)
# Predict the y-values corresponding to x-validation set
y_pred = reg.predict(x_val)

# calculate and print the MAPE and R^2 for the data set
print("Error Data for Test Set Predictions:")
mape = mean_absolute_percentage_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print("MAPE", f"{mape*100:.4f}")
print("R2", f"{r2*100:.4f}")

# Re-calculate the linear regression for the full training dataset
reg.fit(x_data, y_data)
# Read the TEST.csv file and save to x-test and y-test variables
ids, x_test, y_test = dataFromCSV(TEST_FILE)

# Need to drop the categories again, if we want to use them they must be encoded
x_test = x_test.drop(["Cut", "Color", "Clarity", "Symmetry", "Polish", "Report"], axis=1)


# predict the prices for the x-test set
test_pred = reg.predict(x_test)
# convert the array into a pandas Series for data manipulation
test_pred = pd.Series(test_pred)
test_pred = test_pred.rename("Price")

# add the ID's back into the predicted price data and write it to submission.csv
output = ids.to_frame().join(test_pred)
output.to_csv("submission.csv", float_format='%.2f', index=False)
"""
Log:

1) Removed all processing except:
    ordinal encoding of categories and ln(Price)
    -> 5.30% MAPE

2) Added ln(Carat Weight) to processing of x_data -> 5.31% MAPE

3) Drop Carat Weight column -> just marginally worse, added it back in

4) Added "Rarity = Cut + Color + Clarity" -> MAPE = 5.12% test/train, 5.07% online

5) Added scaler -> no change, removed again
6) Added poly -> no change, removed again
7) Added pri_interaction_terms -> 5.16% MAPE test/train

8) Added color_clarity categorical interaction -> 5.09% MAPE

9) Adding cut_color and cut_clarity -> MAPE 5.15%, removed again

10) Carat_bin -> MAPE 5.10%, removed

11) Added secondary interaction onehot terms -> MAPE 5.07% train/test, 4.97% online
12) Added primary interaction terms again, but only carat*color and carat*clarity, got worse, removed

13) Added poly back, MAPE 5.06185, kept

==> Added back secondary regression for Price per carat with a MAPE of 4.85% predicting PPC

14) with secondary regression added -> MAPE 5.0416%

15) Added multiplication interation of PPC * Carat Weight in preprocessor for primary 
    -> MAPE of 4.8526% / 4.8087 online

16) Added primary interactions back of Color*Carat, etc -> MAPE 4.8480% 

17) Increaed n_estimators to 600 -> MAPE 4.7936 train/test, took forever but did reduce online MAPE also

18) Added symmetry_polish categorical one-hot -> MAPE of 4.8163% 
 ---> with n=600 => 4.7741% / 4.7977 online

19) added normalized color = color/carats -> MAPE of 4.8121% 
20) Added Color x Clarity -> 4.8080% 
21) ADded color x cut -> 4.8044%

22) Changed to XGBRegressor for a MAPE of 4.485% <<<<< Benchmark

Outliers remove: 
189, 1882, 811, 47, 4258, 3682, 5990
 -> Public score better, private worse

"""
import xgboost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer

import numpy as np
import csv
import math

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

#########################
### GLOBAL DATA ITEMS ###
#########################
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TEST_SAMPLE = 0.1

# Ordinal Orders
cut = ['Signature-Ideal', 'Ideal', 'Very Good','Good','Fair','Poor']
color = ["D", "E", "F", "G", "H", "I"]
clarity = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1']
sym = ['ID', 'EX', 'VG', 'G']
polish = ['ID', 'EX', 'VG', 'G']
report = ["GIA", "AGSL"]

categories = [cut, color, clarity, sym, polish, report]
categorical_columns = ["Cut", "Color", "Clarity", "Polish", "Symmetry", "Report"] # Ordinals
# one_hot_cols = ["color_clarity", "carat_bin", "cut_clarity", "sym_polish", "cut_color"]
one_hot_cols = ["color_clarity", "sym_polish"]
fxn_cols = ["Carat Weight", "Cut", "Color", "Clarity", "Polish", "Symmetry", "Report"] # Ordinals

def sec_rarity(x):
    ordinal_encoder = OrdinalEncoder(categories=categories)
    x[categorical_columns] = ordinal_encoder.fit_transform(x[categorical_columns])
    x["Rarity"] = x["Color"] + x["Clarity"] + x["Cut"]
    return x

def sec_interaction_terms(x):
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(x[["Cut", "Color", "Clarity"]])
    encoded_cat = encoder.fit_transform(x[["Cut", "Color", "Clarity"]])
    # Get the names of the one-hot encoded columns
    encoded_categorical_cols = encoder.get_feature_names_out(["Cut", "Color", "Clarity"])
    # Create a DataFrame for the one-hot encoded columns
    encoded_df = pd.DataFrame(encoded_cat, columns=encoded_categorical_cols)

    # Combine the encoded columns with the original dataframe
    int_names = []
    for col in encoded_categorical_cols:
        int_names.append(f'{col}_carat_interaction')
        encoded_df[f'{col}_carat_interaction'] = x['Carat Weight'] * encoded_df[col]
    return int_names, encoded_df[int_names]

def pri_interaction_terms(x):
    ordinal_encoder = OrdinalEncoder(categories=categories)
    x[categorical_columns] = ordinal_encoder.fit_transform(x[categorical_columns])
    
    x["normalized_color"] = x["Color"] / x["Carat Weight"]
    x["normalized_clarity"] = x["Clarity"] / x["Carat Weight"]
    x["normalized_polish"] = x["Polish"] / x["Carat Weight"]

    x["Rarity"] = x["Color"] + x["Clarity"] + x["Cut"]
    x["ColorXClarity"] = x["Color"] + x["Clarity"]
    x["ColorXCut"] = x["Color"] + x["Cut"]

    return x

def ppc_carat_int(x):
    x["ppc_x_carat"] = x["Carat Weight"] * x["PPC"]
    return x

######################################
### INITIALIZE PROCESSOR PIPELINES ###
######################################
# rf = RandomForestRegressor(random_state=42, n_estimators=100)
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Secondary pipeline
sec_preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(categories=categories), categorical_columns),
        ('onehot', OneHotEncoder(drop="first", handle_unknown='infrequent_if_exist'), one_hot_cols),
        ('rarity', FunctionTransformer(sec_rarity, validate=False), fxn_cols)
    ], remainder="passthrough", n_jobs=-1
)

pri_preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(categories=categories), categorical_columns),
        ('onehot', OneHotEncoder(drop="first", handle_unknown='infrequent_if_exist'), one_hot_cols),
        ('poly', Pipeline(steps=[('poly', PolynomialFeatures(degree=3, include_bias=True))]), ['ln_carat']),
        ('interaction', FunctionTransformer(pri_interaction_terms, validate=False), fxn_cols),
        ('ppc_inter', FunctionTransformer(ppc_carat_int, validate=False), ["Carat Weight", "PPC"])
    ], remainder="passthrough", n_jobs=-1
)

sec_model = Pipeline([
('preprocessing', sec_preprocessor),
('model', TransformedTargetRegressor(
    regressor=xgb,
    func=np.log,
    inverse_func=np.exp
    ))
])

pri_model = Pipeline([
('preprocessing', pri_preprocessor),
('model', TransformedTargetRegressor(
    regressor=xgb,
    func=np.log,
    inverse_func=np.exp
    ))
])

################################
### FUNCTIONS FOR PROCESSING ###
################################
def binToText(b):
    bin_names = {0:"twenty", 1:"fourty", 2:"sixty", 3:"eighty", 4:"hundred"}
    return bin_names[b]

def getCaratBins(c):
    bins = [np.percentile(c,i) for i in [20, 40, 60, 80]]
    bin_names = ["twenty", "fourty", "sixty", "eighty", "hundred"]
    binned_data = pd.Series(np.digitize(c, bins))
    bin_series = binned_data.map(binToText)
    return bin_series

def dataFromCSV(fname):
    # LOAD DATASET INTO DATAFRAMES
    df = pd.read_csv(fname)
    x_data = df.copy().drop(["ID", "Price"], axis=1)
    y_data = df["Price"]
    ids = df["ID"]
    return ids, x_data, y_data

def prepXData(x_df):
    # Apply the ln function to carat weight and price
    x_df["ln_carat"] = x_df["Carat Weight"].map(np.log) 
    # Generate a combo category of color and clarity together
    x_df['color_clarity'] = x_df['Color'] + "_" + x_df['Clarity']
    x_df["sym_polish"] = x_df["Symmetry"] + "_" + x_df["Polish"]

    int_names, int_df = sec_interaction_terms(x_df[fxn_cols])
    x_df[int_names] = int_df
    return x_df

def getPPC(x_df, y_df):
    return [y_df[index]/row["Carat Weight"] for index, row in x_df.iterrows()]

def getErrorData(actual, pred):
    mape = mean_absolute_percentage_error(actual, pred)
    r2 = r2_score(actual, pred)
    print("MAPE", f"{mape*100:.4f}")
    print("R2", f"{r2*100:.4f}")
    print()

def RunTrainTest(x_data, y_data):
    x_data = prepXData(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=TEST_SAMPLE, random_state=42)

    # Generate the price per carat function
    ppc_y = getPPC(x_train, y_train)

    # Fit the secondary model to the PPC data for the x_train set
    sec_model.fit(x_train, ppc_y)
    # Validate training of secondary model:
    testPPCPred = True
    if(testPPCPred):
        ppc_actual = getPPC(x_test, y_test)
        ppc_pred = sec_model.predict(x_test)
        print("Error Data for PPC Predictions:")
        getErrorData(ppc_actual, ppc_pred)

    # Add predicted PPC to the training model
    x_train["PPC"] = ppc_y

    # Fit the primary model to the training dataset
    pri_model.fit(x_train, y_train)    

    # Predict the ppc for the testing data
    ppc_pred = sec_model.predict(x_test)
    x_test["PPC"] = ppc_pred

    y_pred = pri_model.predict(x_test)

    print("Error Data for Test Set Predictions:")
    getErrorData(y_test, y_pred)

def RunActualData(x_data, y_data):
    x_data = prepXData(x_data)

    print("Loading Data...")
    ppc = getPPC(x_data, y_data)
    # fit the secondary model with all_x data and the ppc data calculated
    print("Fitting secondary model...")
    sec_model.fit(x_data, ppc)
    # now that the model is trained, add the ppc data to the x_data set
    x_data["PPC"] = ppc

    print("Fitting Primary Model...")
    # fit the primary model
    pri_model.fit(x_data, y_data)

    ids, x_test, y_test = dataFromCSV(TEST_FILE)
    x_test = prepXData(x_test)

    train_col = x_data.columns
    test_col = x_test.columns
    for c in train_col:
        if c not in test_col:
            x_test[c] = pd.Series([0 for x in range(len(x_test.index))])

    # Predict and Store PPC values in the test dataset
    print("Predicting secondary data points...")
    ppc_pred = sec_model.predict(x_test)
    x_test["PPC"] = ppc_pred

    print("Predicting primary data points...")
    test_pred = pri_model.predict(x_test)
    test_pred = pd.Series(test_pred)
    test_pred = test_pred.rename("Price")

    output = ids.to_frame().join(test_pred)
    print("Writing output...")
    output.to_csv("submission.csv", float_format='%.2f', index=False)


############################
### MAIN SCRIPT TIMELINE ###
############################
if __name__ == '__main__':
    ids, x_data, y_data = dataFromCSV(TRAIN_FILE)
    RunTrainTest(x_data, y_data)
    # RunActualData(x_data, y_data)
        







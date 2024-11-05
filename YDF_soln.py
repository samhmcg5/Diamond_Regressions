import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import ydf
ydf.verbose(0)

# Ordinal Orders
cat_cols = {
    "Cut":['Signature-Ideal', 'Ideal', 'Very Good','Good','Fair','Poor'],
    "Color":["D", "E", "F", "G", "H", "I"],
    "Clarity":['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1'],
    "Symmetry":['ID', 'EX', 'VG', 'G'],
    "Polish":['ID', 'EX', 'VG', 'G']
}


def getErrorData(actual, pred):
    print("MAPE", f"{mean_absolute_percentage_error(actual, pred):.4f}")
    print("R2", f"{r2_score(actual, pred):.4f}")
    print()

train_file_path = "./train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))

dataset_df = dataset_df.drop('ID', axis=1)

def processForVIz(dataset):
    ds2 = dataset.copy()
    for key in cat_cols.keys():
        ds2[key] = ds2[key].map(cat_cols[key].index)
        dataset[key] = ds2[key].map(str) + '_' + dataset[key]
    return dataset

def processX(dataset):
    dataset["sym_polish"] = dataset["Symmetry"] + "_" + dataset["Polish"]
    dataset["Color_Clarity"] = dataset["Color"] + "_" + dataset["Clarity"]
    # Ordinal encode all the categories
    for key in cat_cols.keys():
        dataset[key] = dataset[key].map(cat_cols[key].index)

    dataset["normalized_color"] = dataset["Color"] / dataset["Carat Weight"]
    dataset["normalized_clarity"] = dataset["Clarity"] / dataset["Carat Weight"]

    dataset["Rarity"] = (dataset["Color"] * dataset["Clarity"] * dataset["Cut"])

    dataset = dataset.drop(["Color", "Clarity"], axis=1)
    dataset["Carat Weight"] = dataset["Carat Weight"].map(np.log)
    dataset["Price"] = dataset["Price"].map(np.log)
    return dataset

# dataset_df = processX(dataset_df)
dataset_df = processForVIz(dataset_df)

def split_dataset(dataset, test_ratio=0.15):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

label = 'Price'

rf = ydf.GradientBoostedTreesLearner(label="Price",
    task=ydf.Task.REGRESSION, 
    growing_strategy="BEST_FIRST_GLOBAL",
    shrinkage=0.1,
    subsample=0.9,
    max_depth=5,
    num_trees=500,
    categorical_algorithm="CART")

model = rf.train(train_ds_pd)

# analysis = model.analyze_prediction(valid_ds_pd.iloc[:1])
# analysis.to_file("analysis.html")

analysis = model.analyze(valid_ds_pd)
analysis.to_file("analysis.html")

print("Evaluation:")
evaluation = model.evaluate(valid_ds_pd)
print()
print(evaluation)
print(model.variable_importances()["NUM_AS_ROOT"])
print()

print("Predictions:")
predictions = model.predict(valid_ds_pd)
# predictions = pd.Series(predictions).map(np.exp)
bench = valid_ds_pd["Price"]#.map(np.exp)
getErrorData(bench, predictions)




# print("\nRunning on full dataset\n")

# model2 = ydf.GradientBoostedTreesLearner(label="Price",
#   task=ydf.Task.REGRESSION, 
#   growing_strategy="BEST_FIRST_GLOBAL",
#   num_trees=500).train(dataset_df)

# test_file_path = "./test.csv"
# test_data = pd.read_csv(test_file_path)
# ids = test_data.pop('ID')

# test_data = processX(test_data)

# preds = model2.predict(test_data)
# preds = pd.Series(preds).map(np.exp)

# output = pd.DataFrame({'ID': ids, 'Price': preds.squeeze()})

# output.to_csv("./output.csv", float_format='%.2f', index=False)


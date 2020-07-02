import pandas as pd
import os
import numpy as np

test_fraction = 0.2

dataset_df = pd.read_csv('./data/all_opposite_CESM.csv')

num_patients = len(dataset_df['Patient_ID'].unique())
train_patients = num_patients * (1 - test_fraction)

training_df = dataset_df[dataset_df['Patient_ID'] <=train_patients]
testing_df = dataset_df[dataset_df['Patient_ID'] > train_patients]

print(testing_df.head())
training_df.to_csv(os.path.join("./data","training_set.csv"), index=False)

testing_df.to_csv(os.path.join("./data","testing_set.csv"), index=False)
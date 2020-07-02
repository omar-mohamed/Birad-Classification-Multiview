import pandas as pd
import os
import numpy as np

test_fraction = 0.2

dataset_df = pd.read_csv('./data/all_opposite_CESM.csv')
uni=dataset_df['Patient_ID'].unique()

num_patients = len(uni)
train_patients = num_patients * (1 - test_fraction)
train_patients = uni[int(train_patients)]
training_df = dataset_df[dataset_df['Patient_ID'] <=train_patients]
testing_df = dataset_df[dataset_df['Patient_ID'] > train_patients]

print(f"len of training: {len(training_df)}, len of testing: {len(testing_df)}")
training_df.to_csv(os.path.join("./data","training_set.csv"), index=False)

testing_df.to_csv(os.path.join("./data","testing_set.csv"), index=False)
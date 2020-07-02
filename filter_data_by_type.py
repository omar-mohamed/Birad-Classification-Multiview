import pandas as pd
import os
import numpy as np

type = 'CESM' # DM or CESM or both

dataset_df = pd.read_csv('./data/all_data.csv')
if type != 'both':
    dataset_df = dataset_df[dataset_df['Type'] == type]


print(len(dataset_df))

image_ids = []

def search_df(df,col,val):
    return len(df[df[col]==val]) > 0

for _, row in dataset_df.iterrows():
    image_name = row['Image_name']
    patient_id = row['Patient_ID']
    side = row['Side']
    if side == 'R':
        opposite_image = image_name.replace('_R_','_L_')
    else:
        opposite_image = image_name.replace('_L_','_R_')

    if search_df(dataset_df, 'Image_name', opposite_image):
        image_ids.append(image_name)

dataset_df = dataset_df[dataset_df['Image_name'].isin(image_ids)]

print(len(dataset_df))

print(dataset_df.head())

dataset_df.to_csv(os.path.join("./data","all_opposite_CESM.csv"), index=False)





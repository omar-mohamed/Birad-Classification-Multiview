import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

csv_name = 'all_data_path_cc_mlo_is_mal.csv'
dataset_df = pd.read_csv('./data/all_data_pathology.csv')

# csv_name = 'new_test_set_path_cc_mlo.csv'
# dataset_df = pd.read_csv('./data/new_test_set.csv')

def make_dict(dataset_df):
    dict={}
    for column in dataset_df:
        dict[column] = []
    return dict

def add_row(dict,df_row):
    for key in df_row.keys():
        if key not in dict.keys():
            dict[key] = []
    for key in dict.keys():
      if key in df_row.keys():
        dict[key].append(df_row[key])


new_csv = make_dict(dataset_df)
new_csv['is_malignant'] = []
pbar = tqdm(total=len(dataset_df))

for i, row in dataset_df.iterrows():
    add_row(new_csv, row)
    if(row["Pathology Classification/ Follow up"]=='Malignant'):
      new_csv['is_malignant'].append('Malignant')
    else:
      new_csv['is_malignant'].append('Non Malignant')



    pbar.update(1)
pbar.close()

new_csv=pd.DataFrame(new_csv)

new_csv.to_csv(os.path.join("./data",csv_name), index=False)


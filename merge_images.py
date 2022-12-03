import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
import math
csv_name = 'omar_test_set_dm_cm.csv'
dataset_df = pd.read_csv('./data/omar_test_set.csv')



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
        dict[key].append(df_row[key])


new_csv = make_dict(dataset_df)

pbar = tqdm(total=len(dataset_df))

for i, row in dataset_df.iterrows():
    if pd.isnull(row['Image_name']):
      continue
    # print(row['Image_name'])
    image_name = row['Image_name'].strip()
    image_name += '.jpg'
    if 'CM' in image_name:
        dm_name = image_name.replace("CM", "DM")
        if os.path.isfile(f"../../birad_classification/Breast-Cancer-Birads-Classification/data/images_rana_cropped_224/{image_name}") and os.path.isfile(f"../../birad_classification/Breast-Cancer-Birads-Classification/data/images_rana_cropped_224/{dm_name}"):
            row['Image_name_DM'] = os.path.splitext(dm_name)[0]
            add_row(new_csv, row)

        else:
            print(f"Did no find dm for {image_name}")
            continue


    pbar.update(1)
pbar.close()

new_csv=pd.DataFrame(new_csv)

new_csv.to_csv(os.path.join("./data",csv_name), index=False)


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

csv_name = 'all_data_path_cc_mlo.csv'
dataset_df = pd.read_csv('./data/all_data_pathology.csv')

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
    image_name = row['Image_name'].strip()
    image_name += '.jpg'
    if 'CC' in image_name:
        mlo_name = image_name.replace("CC", "MLO")
        if os.path.isfile(f"../../birad_classification/Breast-Cancer-Birads-Classification/data/images_rana_cropped_224/{image_name}") and os.path.isfile(f"../../birad_classification/Breast-Cancer-Birads-Classification/data/images_rana_cropped_224/{mlo_name}"):
            row['Image_name_DM'] = os.path.splitext(mlo_name)[0]
            add_row(new_csv, row)

        else:
            print(f"Did no find MLO for {image_name}")
            continue


    pbar.update(1)
pbar.close()

new_csv=pd.DataFrame(new_csv)

new_csv.to_csv(os.path.join("./data",csv_name), index=False)


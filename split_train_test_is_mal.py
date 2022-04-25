import pandas as pd
import os
import numpy as np

dataset_df = pd.read_csv('./data/all_data_path_dm_cm_is_mal.csv')



test_set_fraction=0.15

shuffle=True

if shuffle:
    dataset_df = dataset_df.sample(frac=1., random_state=np.random.randint(1,100))


def get_sparse_labels_mapping(y):
    labels = np.zeros(y.shape[0],dtype=int)
    class_counts = np.zeros(2,dtype=int)
    mapping = {'Non Malignant': 0, 'Malignant': 1}
    index = 0
    for label in y:
        labels[index] = mapping[label]
        class_counts[mapping[label]] += 1

        index += 1
    return labels,class_counts

def get_sparse_labels(y):
    labels = np.zeros(y.shape[0],dtype=int)
    class_counts = np.zeros(2,dtype=int)
    index = 0
    for label in y:
        label = np.array(str(label).split("$"), dtype=np.int) - 1
        labels[index] = int(np.max(label))
        class_counts[labels[index]] += 1

        index += 1
    return labels,class_counts

def make_dict(dataset_df):
    dict={}
    for column in dataset_df:
        dict[column] = []
    return dict

def add_row(dict,df_row):
    for key in dict.keys():
        dict[key].append(df_row[key])

def split_train_test(dataset_df):
    labels= dataset_df['is_malignant']
    sparse_labels,class_counts=get_sparse_labels_mapping(labels)

    test_fraction_count = (class_counts*test_set_fraction).astype(int)
    print("Number of records for each class: {}".format(class_counts))
    print("Number of records for each class in test set: {}".format(test_fraction_count))

    train_dict = make_dict(dataset_df)
    test_dict = make_dict(dataset_df)
    test_count_so_far = np.zeros(2)
    index=0
    for label in sparse_labels:
        if test_count_so_far[label] < test_fraction_count[label]:
            test_count_so_far[label]+=1
            add_row(test_dict,dataset_df.iloc[index])
        else:
            add_row(train_dict,dataset_df.iloc[index])
        index+=1

    return train_dict,test_dict


train_dict,test_dict = split_train_test(dataset_df)

training_df=pd.DataFrame(train_dict)
testing_df=pd.DataFrame(test_dict)

training_df.to_csv(os.path.join("./data","training_set_path_dm_cm_is_mal.csv"), index=False)

testing_df.to_csv(os.path.join("./data","testing_set_path_dm_cm_is_mal.csv"), index=False)
import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
import imgaug.augmenters as iaa


class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, label_columns, multi_label_classification, class_names, source_image_dir,
                 batch_size=16,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=1):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.label_columns = label_columns
        self.multi_label_classification = multi_label_classification
        self.class_counts=[0]*len(class_names)

        self.current_step = -1
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sides = self.side[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = zip(batch_x_path,batch_sides)
        batch_x = np.asarray([self.load_pair(x_path,side) for x_path, side in batch])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def get_opposite_image(self,image_file, side):
        if side == 'R':
            opposite_image = image_file.replace('_R_', '_L_')
        else:
            opposite_image = image_file.replace('_L_', '_R_')

        return opposite_image

    def load_pair(self,image_file, side):
        op_image_name = self.get_opposite_image(image_file, side)
        image = self.load_image(image_file, self.target_size)
        op_image = self.load_image(op_image_name, self.target_size)
        if side == 'R':
            image = np.fliplr(image)
            op_image = np.fliplr(op_image)
        comb = self.images_h_stack([op_image, image])
        comb = resize(comb, self.target_size)
        return comb

    def images_h_stack(self,images):
        list_images = [np.asarray(i) for i in images]
        imgs_comb = np.hstack(list_images)
        return imgs_comb

    def load_image(self,image_file, target_size=None):
        image_path = os.path.join(self.source_image_dir, image_file)
        # image_array = np.random.randint(low=0, high=255, size=( target_size[0],  target_size[1], 3))

        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = self.crop_image(image_array)

        if target_size is not None:
            image_array = resize(image_array, target_size)
        # if side == 'L':
        #     image_array = np.fliplr(image_array)
        return image_array

    def crop_image(self,img, tol=0, margin=100):
        # img is 2D or 3D image data
        # tol  is tolerance
        mask = img > tol
        if img.ndim == 3:
            mask = mask.all(2)
        m, n = mask.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
        row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
        row_start = max(0, row_start - margin)
        col_start = max(0, col_start - margin)
        return img[row_start:row_end + margin, col_start:col_end + margin]

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        if self.multi_label_classification:
            return self.y[:self.steps * self.batch_size, :]
        else:
            return self.y[:self.steps * self.batch_size]


    def get_class_counts(self):
        return self.class_counts

    def get_sparse_labels(self, y):
        labels = np.zeros(y.shape[0],dtype=int)
        index = 0

        for label in y:
            label = np.array(str(label[0]).split("$"), dtype=np.int) - 1
            labels[index] = int(np.max(label))
            # if labels[index] == 4:
            #     labels[index] = 3
            self.class_counts[labels[index]] += 1

            index += 1
        return labels

    def get_onehot_labels(self, y):
        onehot = np.zeros((y.shape[0], len(self.class_names)))
        index = 0
        for label in y:
            labels = np.array(str(label[0]).split("$"), dtype=np.int) - 1
            onehot[index][labels] = 1
            index += 1
        return onehot

    def convert_labels_to_numbers(self, y):
        if self.multi_label_classification:
            return self.get_onehot_labels(y)
        else:
            return self.get_sparse_labels(y)

    def get_images_names(self):
        return self.image_names

    def get_images_path(self, image_names, patient_ids):
        for i in range(image_names.shape[0]):
            image_names[i] =  image_names[i].strip() + '.jpg'
        return image_names

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path, self.y, self.side, self.image_names = self.get_images_path(df["Image_name"].values,
                                                   df["Patient_ID"].values), self.convert_labels_to_numbers(
            df[self.label_columns].values), df['Side'].values, df['Image_name'].values



    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

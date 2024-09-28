import os
import shutil
import uuid

import pydicom as pydicom
from skimage.transform import resize
from visual_model_selector import ModelFactory

from MultiViewModel import MultiViewModel
from configs import argHandler  # Import the default arguments

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
from PIL import ImageTk, Image
import numpy as np
from gradcam import GradCAM
import cv2

GRADCAM_THRESH = 50
WHITE_THRESH = 85
CONFIDENCE_THRESH = 0.50
ONLY_HIGHLIGHTS = True

FLAGS = argHandler()
FLAGS.setDefaults()
target_size = FLAGS.image_target_size
visual_model = None

def load_model():
    global visual_model

    model_factory = ModelFactory()
    if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
        visual_model = MultiViewModel(FLAGS)
        visual_model.built = True
        visual_model.load_weights(FLAGS.load_model_path)
        if FLAGS.show_model_summary:
            visual_model.summary()
    else:
        visual_model = model_factory.get_model(FLAGS)

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def apply_white_threshold(o, h, thresh):
    o_gray = rgb2gray(o)
    o_60 = o_gray < np.percentile(o_gray, thresh)
    h[o_60] = 0
    return h

def get_heatmap_image(image_path, heatmap, cam):
    original = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    if not ONLY_HIGHLIGHTS:
        heatmap = apply_white_threshold(original, heatmap, WHITE_THRESH)
        oneshot = heatmap >= GRADCAM_THRESH
        # oneshot = remove_corner_highlights(heatmap, oneshot)
        heatmap = oneshot * heatmap
        heatmap[heatmap > 0] = 200
    (heatmap, output) = cam.overlay_heatmap(heatmap, original, alpha=0.5)
    # cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    return output

def save_image_to_tmp_folder(img):
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    jpg_image_name = f"{uuid.uuid4()}.jpg"
    jpg_image_path = os.path.join(tmp_dir, jpg_image_name)

    # Save the converted image in the tmp directory
    img.save(jpg_image_path, "JPEG")
    return jpg_image_path



def convert_png_to_jpg(image_path):
    # Check if the image is in PNG format
    if image_path.lower().endswith(".png"):
        img = Image.open(image_path).convert("RGB")
        image_path = save_image_to_tmp_folder(img)
    return image_path


def convert_dicom_to_jpg(image_path):
    if image_path.lower().endswith(".dcm"):
        # Read the DICOM file
        dicom = pydicom.dcmread(image_path)

        # Extract pixel array (image data) from DICOM
        pixel_array = dicom.pixel_array

        # Normalize the pixel array to 8-bit if necessary
        if pixel_array.dtype != np.uint8:
            pixel_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)

        # Check if the pixel array is 3D and take the middle slice
        if pixel_array.ndim > 2:
            middle_index = pixel_array.shape[0] // 2  # Get the index of the middle slice
            pixel_array = pixel_array[middle_index]  # Take the middle slice

        # Convert the NumPy array to a PIL image
        img = Image.fromarray(pixel_array)

        # Save the image as JPEG in the tmp directory
        image_path = save_image_to_tmp_folder(img)

    return image_path

def load_image(image_path):
    # image_array = np.random.randint(low=0, high=255, size=(target_size[0], target_size[1], 3))

    image = Image.open(image_path)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = resize(image_array, target_size)

    return image_array

def handle_and_convert_image(image_path):
    image_path = convert_png_to_jpg(image_path)
    image_path = convert_dicom_to_jpg(image_path)

    return image_path

def transform_batch_images(batch_x):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    batch_x = (batch_x - imagenet_mean) / imagenet_std
    return batch_x

def prepare_model_input(image_cm_path, image_dm_path):
    batch_x_path = [image_cm_path]
    batch_x_dm_path = [image_dm_path]

    batch_x = np.asarray([load_image(x_path) for x_path in batch_x_path])
    batch_x = transform_batch_images(batch_x)

    batch_x2 = np.asarray([load_image(x_path) for x_path in batch_x_dm_path])
    batch_x2 = transform_batch_images(batch_x2)
    batch_x = np.stack((batch_x, batch_x2), axis=0)
    return batch_x

def predict(input):
    y_hat = visual_model(input, training=False)
    print(y_hat)
    if y_hat[0, 0] > 0.5:
        print('Non malignant')
    else:
        print('Malignant')


# sample_input = prepare_model_input('data/images_rana_cropped_224/P254_L_CM_MLO.jpg','data/images_rana_cropped_224/P254_L_DM_MLO.jpg')
#
# print(sample_input.shape)
#
# predict(sample_input)


# Assuming the rest of your code like MultiViewModel, ModelFactory, and FLAGS is already defined.

class ImagePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Predictor")

        # Bind the window's close event to cleanup_tmp_folder function
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.image_cm_path = None
        self.image_dm_path = None

        # UI components
        self.label_cm = Label(root, text="CM Image")
        self.label_cm.grid(row=0, column=0)

        self.label_dm = Label(root, text="DM Image")
        self.label_dm.grid(row=0, column=1)

        self.img_cm_label = Label(root)
        self.img_cm_label.grid(row=1, column=0)

        self.img_dm_label = Label(root)
        self.img_dm_label.grid(row=1, column=1)

        self.upload_cm_button = Button(root, text="Upload CM Image", command=self.load_cm_image)
        self.upload_cm_button.grid(row=2, column=0)

        self.upload_dm_button = Button(root, text="Upload DM Image", command=self.load_dm_image)
        self.upload_dm_button.grid(row=2, column=1)

        self.predict_button = Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=3, column=0, columnspan=2)

        self.result_label = Label(root, text="", font=("Arial", 16))
        self.result_label.grid(row=4, column=0, columnspan=2)

        self.initializing_label = Label(root, text="Initializing...", font=("Arial", 16), fg="blue")
        self.initializing_label.grid(row=5, column=0, columnspan=2)

        # Adding the small button for showing the research info
        self.info_button = Button(root, text="Info", command=self.show_info_dialog)
        self.info_button.grid(row=6, column=0, columnspan=2)

        # Update the UI immediately to show the initializing label
        self.root.update()

        # Load model in the initializer
        self.initialize_model()

    def initialize_model(self):
        # Load the model
        load_model()

        # Update the UI after loading the model
        self.model_loaded()

    def model_loaded(self):
        # Remove the initializing label after model is loaded
        self.initializing_label.grid_remove()

        # Enable buttons after model is loaded
        self.upload_cm_button.config(state="normal")
        self.upload_dm_button.config(state="normal")
        self.predict_button.config(state="normal")


    def show_info_dialog(self):
        info_window = tk.Toplevel(self.root)
        info_window.title("App Information")

        info_text = tk.Text(info_window, wrap="word", width=60, height=15)
        info_text.insert("1.0", """
        This App is only for research purposes, to showcase the following papers:

        - Helal, Maha, Rana Khaled, Omar Alfarghaly, Omnia Mokhtar, Abeer Elkorany, Aly Fahmy, and Hebatalla El Kassas. "Validation of artificial intelligence contrast mammography in diagnosis of breast cancer: Relationship to histopathological results." European Journal of Radiology 173 (2024): 111392.

        - Khaled, Rana, Maha Helal, Omar Alfarghaly, Omnia Mokhtar, Abeer Elkorany, Hebatalla El Kassas, and Aly Fahmy. "Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research." Scientific data 9, no. 1 (2022): 122.

        This research was done by:
        - Omar Alfarghaly (omar.mohamed.latif@gmail.com)
        - Rana Hussien (r_hkhaled@hotmail.com)
        - Maha Helal (dr.mahahelal@yahoo.com)
        - Aly Fahmy (alyfahmy@gmail.com)
        - Abeer elKornay (a.korani@fci-cu.edu.eg)
        - Hebatalla El Kassas
        - Omnia Mokhtar
        """)
        info_text.config(state="disabled")  # Make the text read-only
        info_text.pack(padx=10, pady=10)


    # Function to clear the tmp folder
    def cleanup_tmp_folder(self):
        tmp_dir = "tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            print("Temporary folder cleared.")

    # Function triggered when closing the window
    def on_closing(self):
        # Confirm dialog before closing
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Clean up tmp folder
            self.cleanup_tmp_folder()
            # Close the window
            self.root.destroy()

    def load_cm_image(self):
        self.image_cm_path = filedialog.askopenfilename(title="Select CM Image",
                                                        filetypes=[("Image Files", "*.jpg *.png *.dcm")])
        if self.image_cm_path:
            # Convert PNG to JPG if needed and save it to the tmp folder
            self.image_cm_path = handle_and_convert_image(self.image_cm_path)
            self.show_image(self.image_cm_path, self.img_cm_label)

    def load_dm_image(self):
        self.image_dm_path = filedialog.askopenfilename(title="Select DM Image",
                                                        filetypes=[("Image Files", "*.jpg *.png *.dcm")])
        if self.image_dm_path:
            # Convert PNG to JPG if needed and save it to the tmp folder
            self.image_dm_path = handle_and_convert_image(self.image_dm_path)
            self.show_image(self.image_dm_path, self.img_dm_label)

    def show_image(self, image_path, label):
        img = Image.open(image_path)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk

    def predict(self):
        if self.image_cm_path and self.image_dm_path:
            # Prepare the input data
            input_data = prepare_model_input(self.image_cm_path, self.image_dm_path)
            print("Input shape:", input_data.shape)

            # Make predictions
            y_hat = visual_model(input_data, training=False)
            predicted_class = 0 if y_hat[0, 0] > 0.5 else 1
            label = f"{FLAGS.classes[predicted_class]}: {y_hat[0][1]:.2f}"

            # Generate Grad-CAM heatmaps
            cam = GradCAM(visual_model, predicted_class)
            heatmap1, heatmap2 = cam.compute_heatmap(input_data)

            # Apply heatmap and create output images
            output_image_cm = get_heatmap_image(self.image_cm_path, heatmap1, cam)
            output_image_dm = get_heatmap_image(self.image_dm_path, heatmap2, cam)

            # Ensure tmp directory exists
            tmp_dir = "tmp"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            # Create unique filenames for the heatmap images
            temp_cm_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_cm.jpg")
            temp_dm_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_dm.jpg")

            # Save the heatmap images in the tmp folder
            cv2.imwrite(temp_cm_path, output_image_cm)
            cv2.imwrite(temp_dm_path, output_image_dm)

            # Update the images in the UI
            self.show_image(temp_cm_path, self.img_cm_label)
            self.show_image(temp_dm_path, self.img_dm_label)

            if y_hat[0, 0] > 0.5:
                self.result_label.config(text=label, fg="green")
            else:
                self.result_label.config(text=label, fg="red")
        else:
            self.result_label.config(text="Please upload both images.", fg="orange")

# Create the GUI window
root = tk.Tk()
app = ImagePredictorApp(root)
root.mainloop()
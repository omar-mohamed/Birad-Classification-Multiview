from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from model_utils import set_gpu_usage, get_multilabel_evaluation_metrics, get_generator, get_evaluation_metrics
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import os
import numpy as np
from gradcam import GradCAM
import cv2
from tqdm import tqdm
from MultiViewModel import MultiViewModel
from tensorflow.keras.models import Model


FLAGS = argHandler()
FLAGS.setDefaults()

GRADCAM_THRESH = 50
WHITE_THRESH = 85
CONFIDENCE_THRESH = 0.50
ONLY_HIGHLIGHTS = True
if ONLY_HIGHLIGHTS:
    WRITE_PATH = os.path.join(FLAGS.save_model_path, 'cam_output')
else:
    WRITE_PATH = os.path.join(FLAGS.save_model_path, f"omar_full_test_set_gradcam_thresh_{GRADCAM_THRESH}_white_thresh_{WHITE_THRESH}")


try:
    os.makedirs(WRITE_PATH)
except:
    print("path already exists")

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = MultiViewModel(FLAGS)
    visual_model.built = True
    visual_model.load_weights(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = MultiViewModel(FLAGS)


FLAGS.batch_size = 1
test_generator = get_generator(FLAGS.test_csv,FLAGS)

images_names, images_names_dm = test_generator.get_images_names()

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def apply_white_threshold(o, h, thresh):
    o_gray = rgb2gray(o)
    o_60 = o_gray < np.percentile(o_gray, thresh)
    h[o_60] = 0
    return h

def write_heatmap_image(image_path, heatmap, image_name):
    image_path = image_path.replace('_224','')
    original = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    if not ONLY_HIGHLIGHTS:
        heatmap = apply_white_threshold(original, heatmap, WHITE_THRESH)
        oneshot = heatmap >= GRADCAM_THRESH
        # oneshot = remove_corner_highlights(heatmap, oneshot)
        heatmap = oneshot * heatmap
        heatmap[heatmap > 0] = 200
    (heatmap, output) = cam.overlay_heatmap(heatmap, original, alpha=0.5)
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(WRITE_PATH,image_name), output)

for batch_i in tqdm(range(test_generator.steps)):
    batch, _ = test_generator.__getitem__(batch_i)
    image_path = os.path.join(FLAGS.image_directory, images_names[batch_i])
    image_path_dm = os.path.join(FLAGS.image_directory, images_names_dm[batch_i])
    preds = visual_model.predict(batch)
    predicted_class = 1 if preds[0][1] >= CONFIDENCE_THRESH else 0
    label = f"{FLAGS.classes[predicted_class]}: {preds[0][1]:.2f}"
    cam = GradCAM(visual_model, predicted_class)
    heatmap1, heatmap2 = cam.compute_heatmap(batch)

    write_heatmap_image(image_path, heatmap1, images_names[batch_i])
    write_heatmap_image(image_path_dm, heatmap2, images_names_dm[batch_i])


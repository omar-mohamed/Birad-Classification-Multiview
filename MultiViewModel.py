import tensorflow as tf
from dense_classifier import get_classifier
import numpy as np
from visual_model_selector import ModelFactory


class MultiViewModel(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, FLAGS):
        super(MultiViewModel, self).__init__()
        model_factory = ModelFactory()
        self.concat_dimension = 1
        self.visual_model1 = model_factory.get_model(FLAGS)
        self.visual_model1._name = "visual_model_1"

        self.visual_model2 = model_factory.get_model(FLAGS)
        self.visual_model2._name = "visual_model_2"
        output_shape = self.visual_model1.layers[-1].output.shape.as_list()
        output_shape[self.concat_dimension] *= 2
        self.classifier = get_classifier(output_shape[1:], FLAGS.multi_label_classification, FLAGS.classifier_layer_sizes,
                                         len(FLAGS.classes))

    def call(self, images):
        cm_images = images[0]
        dm_images = images[1]
        cm_features = self.visual_model1(cm_images)
        dm_features = self.visual_model2(dm_images)
        features = tf.concat([cm_features, dm_features], self.concat_dimension)
        pred = self.classifier(features)
        return pred

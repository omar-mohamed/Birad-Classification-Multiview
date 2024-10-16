from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from model_utils import get_optimizer, get_multilabel_class_weights, get_generator, get_class_weights
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
import os
from augmenter import augmenter
from auroc import MultipleClassAUROC
from MultiViewModel import MultiViewModel
import json
import efficientnet.tfkeras

FLAGS = argHandler()
FLAGS.setDefaults()

model_factory = ModelFactory()


# load training and test set file names

train_generator = get_generator(FLAGS.train_csv,FLAGS, augmenter)
test_generator = get_generator(FLAGS.test_csv, FLAGS)

class_weights = None
if FLAGS.use_class_balancing:
    class_weights = get_class_weights(train_generator.get_class_counts(), FLAGS.positive_weights_multiply)


# load classifier from saved weights or get a new one
training_stats = {}
learning_rate = FLAGS.learning_rate

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    multiview_model = MultiViewModel(FLAGS)
    multiview_model.built = True
    multiview_model.load_weights(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        multiview_model.summary()
    training_stats_file = os.path.join(FLAGS.save_model_path, ".training_stats.json")
    if os.path.isfile(training_stats_file):
        training_stats = json.load(open(training_stats_file))
        learning_rate = training_stats['lr']
        print("Will continue from learning rate: {}".format(learning_rate))
else:
    multiview_model = MultiViewModel(FLAGS)

opt = get_optimizer(FLAGS.optimizer_type, learning_rate)

if FLAGS.multi_label_classification:
    multiview_model.compile(loss='binary_crossentropy', optimizer=opt,
                            metrics=[metrics.BinaryAccuracy(threshold=FLAGS.multilabel_threshold)])
else:
    multiview_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    training_stats_file = {}


try:
    os.makedirs(FLAGS.save_model_path)
except:
    print("path already exists")

with open(os.path.join(FLAGS.save_model_path,'configs.json'), 'w') as fp:
    json.dump(FLAGS, fp, indent=4)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=FLAGS.learning_rate_decay_factor,
                      patience=FLAGS.reduce_lr_patience,
                      verbose=1, mode="min", min_lr=FLAGS.minimum_learning_rate)
]

if FLAGS.multi_label_classification:
    checkpoint = ModelCheckpoint(os.path.join(FLAGS.save_model_path, 'latest_model.hdf5'),
                                 verbose=1)

    auroc = MultipleClassAUROC(
        sequence=test_generator,
        class_names=FLAGS.classes,
        weights_path=os.path.join(FLAGS.save_model_path, 'latest_model.hdf5'),
        output_weights_path=os.path.join(FLAGS.save_model_path, 'best_model.hdf5'),
        confidence_thresh=FLAGS.multilabel_threshold,
        stats=training_stats,
        workers=FLAGS.generator_workers,
    )

    callbacks.extend([checkpoint,auroc])
else:
    checkpoint = ModelCheckpoint(os.path.join(FLAGS.save_model_path, 'best_model.hdf5'), monitor='val_accuracy',
                                 save_best_only=True, save_weights_only=False, mode='max', verbose=1)
    callbacks.extend([CSVLogger(os.path.join(FLAGS.save_model_path,'training_log.csv')), checkpoint])

# class_weights = {0: 1.5, 1: 1.0, 2: 2.0}
print(class_weights)
multiview_model.fit(
    train_generator,
    steps_per_epoch=train_generator.steps,
    epochs=FLAGS.num_epochs,
    validation_data=test_generator,
    validation_steps=test_generator.steps,
    workers=FLAGS.generator_workers,
    callbacks=callbacks,
    max_queue_size=FLAGS.generator_queue_length,
    class_weight=class_weights,
    shuffle=False
)

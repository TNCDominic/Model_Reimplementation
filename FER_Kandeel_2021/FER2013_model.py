import tensorflow as tf
import numpy as np


def train_and_fit(BATCH_SIZE = 128):
    IM_HEIGHT = 48
    IM_WIDTH = 48
    SPLIT_RATIO = 0.1
    DAT_PATH = ".\FER2013\FER_2013\\"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DAT_PATH,
        validation_split = SPLIT_RATIO,
        subset="training",
        color_mode = "grayscale",
        seed = 100,
        image_size = (IM_HEIGHT, IM_WIDTH),
        batch_size = BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DAT_PATH,
        validation_split = SPLIT_RATIO,
        subset="validation",
        color_mode = "grayscale",
        seed = 100,
        image_size = (IM_HEIGHT, IM_WIDTH),
        batch_size = BATCH_SIZE
    )

    fist_model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(6, 3, padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax', kernel_regularizer=tf.keras.regularizers.L1(0.01),
                              activity_regularizer=tf.keras.regularizers.L2(0.01))
    ])

    fist_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    filename='BS=%d.csv'%BATCH_SIZE
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    fist_model.fit(
      train_ds,
      validation_data=test_ds,
      epochs=150,
      callbacks=[history_logger],
    )

batch_sizes = [35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 9, 11]

for batch_size in batch_sizes:
    train_and_fit(batch_size)
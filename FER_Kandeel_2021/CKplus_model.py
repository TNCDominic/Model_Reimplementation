import tensorflow as tf

IM_HEIGHT = 48
IM_WIDTH = 48
SPLIT_RATIO = 0.1
BATCH_SIZE = 32
#DAT_PATH =".\FER2013\FER_2013\\"
DAT_PATH = ".\CKPLUS\CK+48\\"

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
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

#fist_model.summary()
#input()

fist_model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

filename = 'test.csv'
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)


fist_model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=150,
  callbacks=[history_logger]
)

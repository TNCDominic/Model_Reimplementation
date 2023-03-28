import tensorflow as tf

IM_HEIGHT = 48
IM_WIDTH = 48
SPLIT_RATIO = 0.15
BATCH_SIZE = 32
DAT_PATH = ".\JAFFE\JAFFE\\"

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

#for dat in test_ds.take(1):
#    print(dat)
#    input()

first_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(6, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

#first_model.summary()
#input()

first_model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

filename = 'JAFFE_Hist.csv'
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

first_model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=100,
  callbacks=[history_logger]
)


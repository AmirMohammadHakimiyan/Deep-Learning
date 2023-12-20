import tensorflow as tf

data_test = "data_test"
data_pa = "dataset"
train_data = tf.keras.utils.audio_dataset_from_directory(
    directory,
    data_pa,
    shuffle=True,
    validation_split=0.2,
    output_sequence_length=48000,
    ragged=False,
    batch_size=4,
    label_mode="categorical",
    labels="inferred",
    sampling_rate=None,
    seed=60,
    subset="training"

)
validation_data = tf.keras.utils.audio_dataset_from_directory(
    directory,
    data_pa,
    batch_size=4,
    shuffle=True,
    subset="validation",
    output_sequence_length=48000,
    label_mode="categorical",
    labels="inferred",
    sampling_rate=None,
    seed=2,
    ragged=False,
    validation_split=0.2
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=80,strides=16,activation="relu", input_shape=(48000 , 1)),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(32, kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(64, kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(64, kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(74, kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(18,activation="softmax"),
])


model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_data, validation_data=validation_data, epochs=24)



import os
import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
import yaml

# Load configurations from YAML file
with open("config.yaml", "r") as yamlfile:
    config = yaml.safe_load(yamlfile)

# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Load the dataset
file_url = config["file_paths"]["dataset"]
dataframe = pd.read_csv(file_url)

# Split the data into training and validation sets
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

# Convert dataframes to TensorFlow datasets
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("labeled_data")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

# Display a sample input and target from the training dataset
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

# Batch the datasets
train_ds = train_ds.batch(config["model_config"]["batch_size"])
val_ds = val_ds.batch(config["model_config"]["batch_size"])

# Define feature space
feature_space_config = config["feature_space"]
feature_space = FeatureSpace(
    features={
        feature["name"]: (FeatureSpace.string_categorical(num_oov_indices=feature["num_oov_indices"])
                          if feature["type"] == "string"
                          else FeatureSpace.integer_categorical(num_oov_indices=feature["num_oov_indices"]))
        for feature in feature_space_config["features"]
    },
    crosses=[
        FeatureSpace.cross(feature_names=cross["feature_names"], crossing_dim=cross["crossing_dim"])
        for cross in feature_space_config["crosses"]
    ],
    output_mode="concat"
)

# Adapt the custom layer to your data
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(config["model_config"]["dense_layer1_units"], activation="relu")(encoded_features)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(config["model_config"]["dense_layer2_units"], activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(config["model_config"]["dropout_rate"])(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer=config["model_config"]["optimizer"],
    loss=config["model_config"]["loss"],
    metrics=config["model_config"]["metrics"]
)

# Train the model with ProbabilityThresholdStop callback
history = training_model.fit(
    preprocessed_train_ds,
    epochs=config["model_config"]["epochs"],
    validation_data=preprocessed_val_ds,
    verbose=2,
)

# Save the trained and inference models
ICMP_training_model = config["file_paths"]["trained_model"]
training_model.save(ICMP_training_model)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
ICMP_inference_model = config["file_paths"]["inference_model"]
inference_model.save(ICMP_inference_model)

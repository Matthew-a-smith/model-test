import json
import pandas as pd
import tensorflow as tf


# Load the inference model
ICMP_inference_model = "Models/Inferance/NMAP-inference_model.keras"
loaded_model = tf.keras.models.load_model(ICMP_inference_model)

# Function to make prediction on the entire dataset
def make_predictions(data):
    input_data = {name: tf.convert_to_tensor(data[name].values.reshape(-1, 1)) for name in data.columns if name not in ['ip.src', 'ip.dst']}
    predictions = loaded_model.predict(input_data)
    return predictions.flatten()

# Load data from CSV file
csv_file = "tcp_traffic.csv"  # Update with your CSV file path
data = pd.read_csv(csv_file)

# Make predictions for the entire dataset
all_predictions = make_predictions(data)

# Set thresholds for printing and writing to JSON
#print_threshold = 0.75
write_threshold = 0.95

# Prepare output data for predictions above threshold
output_data = []
for index, prediction in enumerate(all_predictions):
    if prediction > write_threshold:
        output_data.append({"Index": index, "Prediction": prediction * 100, "Details": data.iloc[index].to_dict()})
    
# Write predictions with additional information to a JSON file
output_file = "predictions.json"  # Update with your output file path
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print("Predictions above the threshold have been written to", output_file)
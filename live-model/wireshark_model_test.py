import csv
import subprocess
import time
import json
import pandas as pd
import tensorflow as tf

def capture_and_predict():
    tshark_cmd = [
        "tshark",
        "-i", "wlan0",
        "-Y", "tcp",
        "-E", "header=y",
        "-T", "fields",
        "-E", "separator=, ",
        "-e", "ip.dst",
        "-e", "ip.src",
        "-e", "tcp.dstport",
        "-e", "tcp.srcport",
        "-e", "tcp.window_size_value",
        "-e", "tcp.flags",
    ]

    inference_model = "Models/Inferance/NMAP-inference_model.keras"
    loaded_model = tf.keras.models.load_model(inference_model)

    try:
        while True:
            with open("tcp_traffic.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["ip.dst", "ip.src", "tcp.dstport", "tcp.srcport", "tcp.window_size_value", "tcp.flags"])

                # Start capturing TCP traffic and write to CSV
                process = subprocess.Popen(tshark_cmd, stdout=subprocess.PIPE, universal_newlines=True)
                
                try:
                    for line in process.stdout:
                        # Write captured data to CSV
                        writer.writerow(line.strip().split(','))

                        # Read data into DataFrame
                        data = pd.read_csv("tcp_traffic.csv")

                        # Make predictions
                        predictions = make_predictions(data, loaded_model)

                        # Filter predictions
                        high_probability_predictions = [(index, prediction) for index, prediction in enumerate(predictions) if prediction > 0.95]

                        # Prepare output data
                        if not high_probability_predictions:
                            high_probability_predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:20]

                        output_data = [{"Index": index, "Prediction": prediction * 100, "Details": data.iloc[index].to_dict()} for index, prediction in high_probability_predictions]

                        # Write predictions to JSON file
                        with open("predictions.json", 'w') as f:
                            json.dump(output_data, f, indent=4)

                        print("Predictions have been written to predictions.json")

                        time.sleep(5)  # Adjust as needed for the frequency of predictions

                except KeyboardInterrupt:
                    print("\nTCP traffic capture and prediction stopped.")
                    process.kill()
                    break

    except Exception as e:
        print("An error occurred:", e)

def make_predictions(data, model):
    input_data = {name: tf.convert_to_tensor(data[name].values.reshape(-1, 1)) for name in data.columns if name not in ['ip.src', 'ip.dst']}
    predictions = model.predict(input_data)
    return predictions.flatten()

if __name__ == "__main__":
    capture_and_predict()

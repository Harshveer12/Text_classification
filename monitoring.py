# import numpy as np
# import tensorflow as tf
# from prometheus_client import Gauge, start_http_server
# from sklearn.metrics import accuracy_score, f1_score
# from scipy.stats import entropy
# from initializing import Initialize

# training_data = Initialize('aclImdb_v1.tar.gz')

# # Load the trained model
# model = tf.keras.models.load_model('/home/harsh/Documents/Bert_classification/model/zipfolder')

# # Initialize Prometheus metrics
# accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the text classification model')
# f1_gauge = Gauge('model_f1', 'Current F1 score of the model')

# # Start Prometheus HTTP server
# start_http_server(8000)

# # Function to track accuracy and other metrics
# def monitor_model_performance(test_ds):
#     y_true, y_pred = [], []

#     # Generate predictions on the test dataset
#     for text_batch, label_batch in test_ds:
#         print("Here")
#         predictions = model.predict(text_batch)
#         predicted_labels = np.argmax(predictions, axis=1)
#         y_true.extend(label_batch.numpy())
#         y_pred.extend(predicted_labels)

#     # Calculate accuracy and F1 score
#     accuracy = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average='weighted')
    
#     # Update Prometheus metrics
#     accuracy_gauge.set(accuracy)
#     f1_gauge.set(f1)
    
#     print(f"Accuracy: {accuracy}, F1 Score: {f1}")

# # Function to monitor data drift (using KL divergence)
# def check_data_drift(train_dist, live_dist, threshold=0.1):
#     drift = entropy(train_dist, live_dist)
#     if drift > threshold:
#         print(f"Data drift detected: {drift}")
#         trigger_retrain()
#     else:
#         print(f"No significant drift. KL Divergence: {drift}")

# # Trigger retrain if needed
# def trigger_retrain():
#     print("Triggering model retrain...")
#     # Add retrain logic here

# # Example of monitoring function usage
# if __name__ == '__main__':
#     # Assuming 'train_ds' and 'test_ds' are already created using your `make_dataset` method
#     _, _, test_ds = training_data.make_dataset('aclImdb/train', 'aclImdb/test')

#     # Monitor model performance on test dataset
#     monitor_model_performance(test_ds)
    
#     # Assuming you have some method to get distributions for data drift (can use histograms)
#     # Example: train_dist and live_dist from text data
#     # train_dist = [class distribution from training data]
#     # live_dist = [class distribution from live data]
#     # check_data_drift(train_dist, live_dist)


import numpy as np
from prometheus_client import Gauge, start_http_server
from sklearn.metrics import accuracy_score, f1_score
from initializing import Initialize
from retraining import Retrain
import tensorflow as tf

training_data = Initialize('aclImdb_v1.tar.gz')
# Initialize Prometheus metrics
accuracy_gauge = Gauge('model_accuracy', 'Current accuracy of the text classification model')
f1_gauge = Gauge('model_f1', 'Current F1 score of the model')


# Function to track accuracy and other metrics
def monitor_model_performance(test_ds, model_path):
    y_true, y_pred = [], []
    model = tf.keras.models.load_model(model_path)
    # Generate predictions on the test dataset
    for text_batch, label_batch in test_ds:
        print("Here")
        predictions = model.predict(text_batch)
        predicted_labels = np.argmax(predictions, axis=1)
        y_true.extend(label_batch.numpy())
        y_pred.extend(predicted_labels)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Update Prometheus metrics
    accuracy_gauge.set(accuracy)
    f1_gauge.set(f1)
    
    print(f"Accuracy: {accuracy}, F1 Score: {f1}")


# Function to calculate statistics from a dataset
def calculate_statistics(dataset):
    means, variances = [], []
    for text_batch, _ in dataset:
        # Convert text batches to string lengths as a proxy feature
        lengths = [len(text.decode('utf-8')) for text in text_batch.numpy()]
        means.append(np.mean(lengths))
        variances.append(np.var(lengths))

    # Compute overall mean and variance
    overall_mean = np.mean(means)
    overall_variance = np.mean(variances)
    return overall_mean, overall_variance

# Function to monitor data drift using key statistics
def check_data_drift(train_stats, live_stats, threshold=0.1):
    train_mean, train_variance = train_stats
    live_mean, live_variance = live_stats
    
    # Check for significant shift in mean and variance
    mean_shift = abs(train_mean - live_mean) / train_variance
    variance_shift = abs(train_variance - live_variance) / train_variance

    print(f"Mean Shift: {mean_shift}, Variance Shift: {variance_shift}")
    
    # Trigger retrain if drift exceeds the threshold
    if mean_shift > threshold or variance_shift > threshold:
        print(f"Data drift detected. Triggering retrain...")
        trigger_retrain()
    else:
        print("No significant data drift detected.")
# Trigger retrain if needed
def trigger_retrain():
    print("Triggering model retrain...")
    old_dataset_path = 'aclImdb'
    new_dataset_path = '/home/harsh/Documents/new_data'
    combined_dataset_path = 'combined_dataset'

    retrain = Retrain(old_dataset_path,new_dataset_path,combined_dataset_path)
    retrain.retrain_model()
    retrain.save_retrained_model()
    # Call your retrain logic here
# Example of how to run drift monitoring
def run_drift_monitoring(train_ds, test_ds):
    # Calculate statistics for train and test datasets
    print("Calling stats")
    train_stats = calculate_statistics(train_ds)
    live_stats = calculate_statistics(test_ds)
    print("calling check")
    # Check for data drift
    check_data_drift(train_stats, live_stats, threshold=0.1)


# Monitoring in action
print("Running")
train_ds, val_ds, test_ds = training_data.make_dataset('aclImdb/train', '/home/harsh/Documents/combined_dataset/train')
run_drift_monitoring(train_ds, test_ds)

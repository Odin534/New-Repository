import csv
import os

def compare_and_populate_labels(eeg_file, label_file, output_file):
    eeg_data = []
    with open(eeg_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            eeg_data.append(row)
    
    labels_data = []
    with open(label_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            labels_data.append((float(row[0]), row[2]))  # Notice the change here: row[2] is the label
    
    output_data = []
    for eeg_row in eeg_data:
        eeg_timestamp = float(eeg_row[39])  # Assuming the timestamp column is at index 38
        matched_label = None
        for label_row in labels_data:
            if abs(eeg_timestamp - label_row[0]) < 0.01:  # Modify the tolerance as needed
                matched_label = label_row[1]  # The label is now the second item in the tuple
                break
        eeg_row.append(matched_label)  # Add the matched label to the end of the row
        output_data.append(eeg_row)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(output_data)

# File paths
current_directory = os.path.dirname(os.path.abspath(__file__))
eeg_file = os.path.join(current_directory, 'data', 'eeg_data.csv')
label_file = os.path.join(current_directory, 'data.csv')
output_file = os.path.join(current_directory, 'data', 'eeg_data_with_labels.csv')

# Compare timestamps and populate labels
compare_and_populate_labels(eeg_file, label_file, output_file)

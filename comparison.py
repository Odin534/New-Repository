import matplotlib.pyplot as plt
import seaborn as sns

# Redefining the confusion matrices
Normalized_confusion_matrices = {
    "GRU": [[0.56, 0.25, 0.19],
             [0.13, 0.66, 0.21],
            [0.11, 0.27, 0.63]],
    "LSTM": [[0.78, 0.10, 0.12 ],
            [0.15, 0.70, 0.15],
            [0.13, 0.12, 0.75]],
    "FFT-SVM": [[0.52, 0.12, 0.36],
                [0.42, 0.21, 0.37],
                [0.41, 0.12, 0.47]],

    "PSD-SVM": [[0.98, 0.02, 0],
                     [0, 0.95, 0.05],
                     [0.5, 0.5, 0.90]],

    "EEGNET": [[0.51, 0.17, 0.32],
               [0.14, 0.51, 0.35],
               [0.11, 0.12, 0.67]],
    "FFT-KNN": [[0.50, 0.30, 0.20],
                     [0.42, 0.38, 0.20],
                     [0.43, 0.32, 0.25]],
    "DWT-KNN": [[0.99, 0.01, 0],
                     [0, 0.99, 0.01],
                     [0, 0, 1]],
    "DWT-SVM (6 channel)": [[0.97, 0.02, 0.01],
                [0, 0.96, 0.04],
                [0, 0, 1]],
    
    "EEGNET-Bhagyashree": [[0.52,0.16,0.32],
                [ 0.22, 0.46, 0.32],
                [0.11 , 0.23, 0.66]],
    
    "ChronoNet": [[0.89, 0.06, 0.05],
                  [ 0.08, 0.80, 0.12],
                  [ 0.06, 0.07, 0.87]]

    
}

labels = ["Left", "Right", "Stop"]

# Plotting the heatmaps for confusion matrices
plt.figure(figsize=(24, 24))

plt.figure(figsize=(24, 24))  # Adjust the figure size to give more space

for idx, (model, matrix) in enumerate(Normalized_confusion_matrices.items(), 1):
    plt.subplot(4, 3, idx)  # Change this line to a 4x3 layout
    sns.heatmap(matrix, annot=True, fmt='g', cmap="YlGnBu", cbar=False, xticklabels=labels, yticklabels=labels, annot_kws={"size": 10})
    plt.title(f'{model}', fontsize=16)
    plt.ylabel('Actual Labels', fontsize=10)
    plt.xlabel('Predicted Labels', fontsize=10)
    plt.suptitle("Normalized Confusion Matrix", fontsize=18)

plt.tight_layout(pad=3.0)
plt.gcf().canvas.setWindowTitle("Normalized Confusion Matrix")
plt.show()

import pandas as pd

tpr_data = {}
for model, matrix in Normalized_confusion_matrices.items():
    tpr_data[model] = [matrix[i][i] / sum(matrix[i]) for i in range(3)]
    
df_tpr = pd.DataFrame(tpr_data, index=labels)

# Creating a DataFrame for TPR and plotting the bar chart
df_tpr.plot(kind="bar", figsize=(18, 10))
plt.title("True Positive Rate (TPR) Comparison", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=14)
plt.xlabel("Classes", fontsize=14)
plt.ylim(0, 1)
plt.legend(title="Models", fontsize=12)
plt.suptitle("TPR of Normalized Confusion Matrix", fontsize=18)
plt.grid(axis='y')
plt.tight_layout()
plt.gcf().canvas.setWindowTitle("TPR of Normalized Confusion Matrix")
plt.show()

#The True Positive Rate (also known as Recall or Sensitivity) for each class is given by:

#TPR (for a class) = Number of True Positives (for that class) / Total Actual (for that class)

# Compute overall accuracies for each model
overall_accuracies = {}
for model, matrix in Normalized_confusion_matrices.items():
    true_positives = sum([matrix[i][i] for i in range(3)])
    overall_accuracies[model] = true_positives / 3

# Convert the accuracies to a DataFrame
df_accuracies = pd.DataFrame(list(overall_accuracies.items()), columns=['Model', 'Accuracy'])

# Sort the DataFrame by accuracy
df_accuracies = df_accuracies.sort_values(by='Accuracy', ascending=False)

# Plot the overall accuracies
plt.figure(figsize=(18, 10))
sns.barplot(x='Model', y='Accuracy', data=df_accuracies, palette='viridis')
plt.title('Overall Accuracy Comparison', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Models', fontsize=14)
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

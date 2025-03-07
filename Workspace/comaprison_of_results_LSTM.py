import matplotlib.pyplot as plt

models = [
    "Occipetal & Parietal EEG Channels",
    "LSTM using 1s window (5 files of Sujay)",
    "LSTM without window (5 files of Sujay)",
    "LSTM without window using all datasets",
    "LSTM with 1s window using all datasets",
    "PSD + LSTM + 1s window",
    "DWT + LSTM + 1s window",
    "FFT + LSTM + 1s window"
]

test_accuracy = [59, 73, 74, 64, 62, 76, 25, 26.78]

plt.figure(figsize=(10, 6))
plt.bar(models, test_accuracy, color='darkblue')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Comparison- LSTM')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
